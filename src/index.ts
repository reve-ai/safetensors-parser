// Bytes is the byte array type (Uint8Array)
export type Bytes = Uint8Array;
// Format is typically BF16 or FP16 or somesuch, but it's not really checked
// or used by the safetensors library itself -- it just deals with byte arrays.
export type Format = string;
// The only numbers we deal with are integers, so document that.
export type Integer = number;
// Offset pair is [start, end) of the array sub-range.
export type OffsetPair = [Integer, Integer];
// Shape is an n-dimensional array of integers.
export type Shape = Integer[];
// Tensor names are strings.
export type TensorName = string;

export class IgnorableError extends Error {
	constructor(message: string) {
		super(message + " You can ignore this error with ignoreInvalid=true in parseSafeTensors().");
		this.name = "IgnorableError";
	}
}

// Given a safetensors file (as a Uint8Array) return the TensorMap of the
// tensors and metadata stored in that file. If the file is not fully compliant
// with the spec (additional properties, etc) an error will be thrown. If you'd
// like to try using it anyway, you can pass ignoreInvalid=true.
export function parseSafeTensors(bytes: Bytes, ignoreInvalid?: boolean): TensorMap {
	sanityCheckTensorsHeader(!!ignoreInvalid, bytes, bytes.length);
    const hdrsize = unsafeGetHeaderSize(bytes);
    const ret = new TensorMap();
    const j: Object = JSON.parse(new TextDecoder().decode(bytes.slice(8, 8+hdrsize)));
	sanityCheckTensorsParsed(!!ignoreInvalid, j, bytes.length - 8 - hdrsize);
    for (const [name, val] of Object.entries(j)) {
        if (name === "__metadata__") {
            ret.setAllMetadata(new Map<string, string>(Object.entries(val as Object)));
        } else {
            const {dtype, shape, data_offsets} = val as { dtype: string, shape: Shape, data_offsets: OffsetPair };
            ret.addTensor(name, bytes.slice(8+hdrsize+data_offsets[0], 8+hdrsize+data_offsets[1]), dtype, shape);
        }
    }
    return ret;
}

// Given a TensorMap, generate the contents of a safetensors file that contains
// those tensors, and the added metadata. This will throw if you run out of memory,
// OR if you try to save a tensor with metadata that is too large. There's a
// maximum header/metadata limit of 100 MB total in the file format spec.
// You can write by passing in an empty Uint8Array, or you can pass in a function
// that will be called with chunks of data in order of writing.
// This implementation will align the start of each tensor on a multiple of 8 bytes
// if it's less than that.
export function saveSafeTensors(tensorMap: TensorMap): Uint8Array
export function saveSafeTensors(tensorMap: TensorMap, write: (data: Bytes) => void): undefined
export function saveSafeTensors(tensorMap: TensorMap, write?: (data: Bytes) => void): Uint8Array | undefined {
	const save = new TensorSaver(tensorMap, write);
	save.calcOffset();
	save.calcHeader();
	save.writeHeader();
	save.writeTensors();
	return save.ret;
}

const padder = new Uint8Array([32, 32, 32, 32, 32, 32, 32, 32]);

// Simple helper to format the safetensors file with header and data chunks.
class TensorSaver {
	readonly hdr: { [key: string]: {dtype: string, shape: Shape, data_offsets: OffsetPair} } & { "__metadata__"?: {[key: string]: string} } = {};
	constructor(readonly tensorMap: TensorMap, private write?: (data: Bytes) => void) {
		this.setMetadata();
	}
	tes?: Array<[string, TensorRef]>;
	offset?: Integer;
	hdrBuf?: Uint8Array;
	lenbuf?: Uint8Array;
	hblen?: Integer;
	ret?: Uint8Array;
	setMetadata() {
		const md = {} as {[key: string]: string};
		let toset =  false;
		for (const [name, val] of this.tensorMap.allMetadata.entries()) {
			md[name] = val;
			toset = true;
		}
		if (toset) {
			this.hdr["__metadata__"] = md;
		}
	}
	calcOffset(): Integer {
		let offset = 0;
		this.tes = Array.from(this.tensorMap.allTensors.entries());
		for (const [name, tensor] of this.tes) {
			this.hdr[name] = {
				dtype: tensor.format,
				shape: tensor.shape,
				data_offsets: [offset, offset + tensor.bytes.length]
			};
			offset += (tensor.bytes.length + 7) & ~7;
		}
		this.offset = offset;
		return offset;
	}
	calcHeader(): Integer {
		const hdrBuf = (new TextEncoder()).encode(JSON.stringify(this.hdr));
		const hblen = (hdrBuf.length + 7) & ~7;
		if (hblen > 100*1024*1024) {
			throw new Error("The metadata is too large to be saved in a safetensors file.");
		}
		this.lenbuf = new Uint8Array([hblen & 0xff, (hblen >> 8) & 0xff, (hblen >> 16) & 0xff, (hblen >> 24) & 0xff, 0, 0, 0, 0]);
		this.hdrBuf = hdrBuf;
		this.hblen = hblen;
		return hblen;
	}
	writeHeader() {
		if (!this.write) {
			this.ret = new Uint8Array(this.offset! +this. hblen! + 8);
			let wrote = 0;
			this.write = (data: Bytes) => {
				this.ret!.set(data, wrote);
				wrote += data.length;
			};
		}
		this.write(this.lenbuf!);
		this.write(this.hdrBuf!);
		if (this.hblen! > this.hdrBuf!.length) {
			this.write(padder.slice(0, this.hblen! - this.hdrBuf!.length));
		}
	}
	writeTensors() {
		let written = 0;
		for (const [name, tensor] of this.tes!) {
			this.write!(tensor.bytes);
			written += tensor.bytes.length;
			if (tensor.bytes.length & 7) {
				this.write!(padder.slice(0, 8 - (tensor.bytes.length & 7)));
				written += 8 - (tensor.bytes.length & 7);
			}
			const end: Integer = (this.hdr[name] as {data_offsets: OffsetPair})["data_offsets"][1];
			if (((end + 7) & ~7) !== written) {
				throw new Error(`Internal tensor alignment problem: "${name}": ${end} !== ${written}`);
			}
		}
	}
}

// TensorMap is a collection of possibly multiple tensors. It has either been manually
// constructed, or loaded from a safetensors file. It can be used to return data (in
// little endian format) or to save the tensors to a safetensors formatted file. Note
// that tensor refs are not cloned -- if you mutate the underlying data, you will
// indirectly mutate the TensorMap, too!
export class TensorMap {
	refs: Map<string, TensorRef> = new Map();
	metadata: Map<string, string> = new Map();
	constructor() {}
	// If the tensor exists in the map, return it, else return undefined.
	getTensor(name: TensorName): TensorRef | undefined {
		return this.refs.get(name);
	}

	// Add a new tensor to the map. If another tensor with the same name already exists,
	// fail. Use setTensor() if you don't care.
	addTensor(name: TensorName, bytes: Bytes, format: Format, shape: Shape): TensorRef
	addTensor(name: TensorName, tensor: TensorRef): TensorRef
	addTensor(name: TensorName, data: TensorRef | Bytes, format?: Format, shape?: Shape): TensorRef {
		if (this.refs.get(name)) {
			throw new Error(`The name "${name}" already exists in the TensorMap.`);
		}
		if (data instanceof TensorRef) {
			return this._setTensor(name, data);
		}
		if (!format || !shape) {
			throw new Error(`You must provide format and shape when adding tensor "${name}".`);
		} else {
			return this._setTensor(name, new TensorRef(name, data, format, shape));
		}
	}

	// Set this name to be this tensor, no matter whether it exists or not. Remove
	// any previous tensor of the same name (which may be the same value!)
	setTensor(name: TensorName, tensor: TensorRef): TensorRef {
		this.removeTensor(name);
		return this.addTensor(name, tensor);
	}

	// If the tensor exists, then return it. Otherwise, create it with the factory
	// function, add it to the map, and return it.
	getOrMakeTensor(name: TensorName, factory: () => TensorRef): TensorRef {
		let ten = this.refs.get(name);
		if (ten) {
			return ten;
		}
		ten = factory();
		this.refs.set(name, ten);
		return ten;
	}

	// If some metadata value exists, return it, else return undefined.
	getMetaValue(name: TensorName): string | undefined {
		return this.metadata.get(name);
	}

    setMetaValue(name: TensorName, value: string) {
        this.metadata.set(name, value);
    }

	get allMetadata(): Map<string, string> {
		return this.metadata;
	}

	get allTensors(): Map<string, TensorRef> {
		return this.refs;
	}

    setAllMetadata(metadata: Map<string, string>) {
        this.metadata = metadata;
    }

	// Remove a tensor from the map. It's OK if it doesn't exist.
	removeTensor(name: TensorName | TensorRef) {
		let ten: TensorRef | undefined;
		if (typeof name === "string") {
			ten = this.refs.get(name);
			if (!ten) {
				return;
			}
		} else {
			ten = name;
		}
		if (ten.parent !== this) {
			throw new Error("You can only remove a tensor from the TensorMap that it belongs to.");
		}
		delete ten._parent;
		this.refs.delete(ten.name);
	}

	private _setTensor(name: TensorName, tensor: TensorRef): TensorRef {
		if (tensor.parent && tensor.parent !== this) {
			throw new Error("You can only add a tensor to a TensorMap that it doesn't already belong to.");
		}
		this.refs.set(name, tensor);
		tensor._parent = this;
		return tensor;
	}
};

// TensorRef is a specific tensor within a safetensors archive. You can make one free
// standing, or get one from an archive.
export class TensorRef {
	// Make a new tensorref referencing the given data. The new tensorref is not parented.
	// Note that the shape and format are not sanity checked! Hopefully the bytes are accurate.
	constructor(name: TensorName, readonly bytes: Bytes, readonly format: Format, readonly shape: Shape) {
		this._name = name;
	}
	// A tensorref may have a parent tensor map, or may be freestanding.
	get parent(): TensorMap | undefined { return this._parent; }
	private _name: TensorName;
	_parent?: TensorMap;
	// If you mutate the name of a tensor in a tensor map, it will rename the tensor
	// in the map. It's an error to rename to a name that already exists in the map.
	get name(): string { return this._name; }
	set name(val: string) {
		if (this._parent) {
			if (this._parent.refs.get(val)) {
				throw new Error(`The name ${val} already exists in the parent TensorMap.`);
			}
			this._parent.refs.delete(this._name);
		}
		this._name = val;
		if (this._parent) {
			this._parent.refs.set(this._name, this);
		}
	}
	// If the tensor is parented, remove it from the parent.
	removeIfParented() {
		if (this._parent) {
			this._parent.removeTensor(this);
		}
	}
	// Check whether the tensor is sane
	sanityCheck() {
		sanityCheck(this);
	}
};


// Given the bytes of a safetensors file, verify that the header seems legitimate.
// Note that you only need 10 bytes to check this. The value returned is the size
// of the JSON chunk that starts at offset 8, so if the total file size is smaller
// than 8+return, the file is likely truncated.
export function sanityCheckTensorsHeader(ignoreInvalid: boolean, bytes: Bytes, filesize: Integer): Integer {
    if (!arrayCompare(Array.from(bytes.slice(4,9)), [0, 0, 0, 0, "{".charCodeAt(0)])) {
        throw new Error("The file header is not a valid safetensors file.");
    }
    const val = unsafeGetHeaderSize(bytes);
    if (val > 100*1024*1024 && !ignoreInvalid) {
        throw new IgnorableError("The file header is too long to be a valid safetensors file.");
    }
    if (filesize < 8+val) {
        throw new Error("The safetensors file seems truncated or otherwise not valid.");
    }
    return val;
}

export function unsafeGetHeaderSize(bytes: Bytes): Integer {
    return ((bytes[0]! | 0) + ((bytes[1]! | 0) * 256) + ((bytes[2]! | 0) * 256 * 256) + ((bytes[3]! | 0) * 256 * 256 * 256)) | 0;
}

// Return true if each element compares equal in the arrays
function arrayCompare(a: Shape, b: Shape): boolean {
    if (a.length !== b.length) {
        return false;
    }
    for (let i = 0; i < a.length; i++) {
        if (a[i] !== b[i]) {
            return false;
        }
    }
    return true;
}

export function sanityCheckTensorsParsed(ignoreInvalid: boolean, j: Object, chunksize: Integer): void {
    const covered: OffsetPair[] = [];
    for (const [name, value] of Object.entries(j)) {
        if (name === "__metadata__") {
            checkMetadata(ignoreInvalid, value);
            continue;
        }
		const tac = new TensorAttributeChecker(chunksize, covered, ignoreInvalid);
        tac.checkTensorValue(name, value);
    }
}

function checkMetadata(ignoreInvalid: boolean, value: Object): void {
    for (const [key, strval] of Object.entries(value)) {
        if (typeof strval !== "string" && !ignoreInvalid) {
            throw new IgnorableError(`The metadata value ${key} is not a string (${typeof strval}).`);
        }
    }
}

class TensorAttributeChecker {
    constructor(readonly chunksize: Integer, readonly covered: OffsetPair[], readonly ignoreInvalid: boolean = false, public has_dtype: boolean = false, public has_shape: boolean = false, public has_data_offsets: boolean = false) {}
    dtype(key: string, val: any) {
        if (typeof val !== "string") {
            throw new Error(`The dtype for ${key} is not a string (${typeof val}).`);
        }
        this.has_dtype = true;
    }
    shape(key: string, val: any) {
        this.checkTensorShape(key, val);
        this.has_shape = true;
    }
    data_offsets(key: string, val: any) {
        this.checkDataOffsets(key, val);
        this.has_data_offsets = true;
    }
    attrOk(key: string): boolean {
        return key === "dtype" || key === "shape" || key === "data_offsets";
    }
    assertAttributePresence(name: TensorName) {
        if (!this.has_dtype) {
            throw new Error(`The tensor "${name}" is missing the dtype key.`);
        }
        if (!this.has_shape) {
            throw new Error(`The tensor "${name}" is missing the shape key.`);
        }
        if (!this.has_data_offsets) {
            throw new Error(`The tensor "${name}" is missing the data_offsets key.`);
        }
    }
    private checkTensorShape(key: string, val: Object): void {
        if (!Array.isArray(val)) {
            throw new Error(`The shape for ${key} is not an array (${typeof val})`);
        }
        for (const v of val) {
            if (typeof v !== "number" && !this.ignoreInvalid) {
                throw new IgnorableError(`The shape for ${key} is not an array of numbers (${typeof v})`);
            }
        }
    }
    private checkDataOffsets(key: string, obj: Object): void {
		if (!Array.isArray(obj) || obj.length !== 2) {
			throw new Error(`The data_offsets for ${key} is not an array with length 2 (${typeof obj})`);
		}
		this.checkDataOffsetBasics(key, obj as OffsetPair);
        for (const [start, end] of this.covered) {
            this.checkTensorOverlap(key, start, end, obj as OffsetPair);
        }
        this.covered.push(obj as OffsetPair);
    }
	checkDataOffsetBasics(key: string, val: OffsetPair) {
		let err: Error | null = null;
		if (!pairInRange(val, this.chunksize) && !this.ignoreInvalid) {
			err = new IgnorableError(`The data_offsets for ${key} is out of range (${val[0]}-${val[1]} versus ${this.chunksize}).`);
		}
		if (err && !this.ignoreInvalid) {
			throw err;
		}
	}
	checkTensorOverlap(key: string, start: Integer, end: Integer, val: OffsetPair): void {
		let err: Error | null = null;
		if (val[1] > start && val[1] <= end) {
			err = new IgnorableError(`The data_offsets end for ${key} overlaps with another tensor (${val[1]} > ${start} && ${val[1]} <= ${end}).`);
		}
		if (val[0] >= start && val[0] < end) {
			err = new IgnorableError(`The data_offsets start for ${key} overlaps with another tensor (${val[0]} >= ${start} && ${val[0]} < ${end}).`);
		}
		if (err && !this.ignoreInvalid) {
			throw err;
		}
	}
	checkTensorValue(name: TensorName, value: Object): void {
		for (const [key, val] of Object.entries(value)) {
			if (!this.attrOk(key)) {
				if (!this.ignoreInvalid) {
					throw new IgnorableError(`The tensor description for "${name}" has an invalid key "${key}".`);
				}
				continue;
			}
			(this[key as keyof TensorAttributeChecker] as (s: string, u: unknown) => void)(name, val);
		}
		this.assertAttributePresence(name);
	}
}

function pairInRange(pair: OffsetPair, chunksize: Integer): boolean {
	return pair[0] >= 0 && pair[0] <= chunksize && pair[1] >= 0 && pair[1] <= chunksize && pair[0] <= pair[1];
}

export function sanityCheck(tensor: TensorRef): void
export function sanityCheck(bytes: Bytes, format: Format, shape: Shape): void
export function sanityCheck(bytes: TensorRef | Bytes, format?: Format, shape?: Shape): void {
	if (bytes instanceof TensorRef) {
		_sanityCheck(bytes.name, bytes.bytes, bytes.format, bytes.shape);
	} else {
		_sanityCheck("created", bytes, format, shape)
	}
}

function _sanityCheck(name: TensorName, bytes: Bytes, format?: Format, shape?: Shape) {
	if (!format || !shape) {
		throw new Error(`The tensor "${name}" is missing format and shape information.`);
	}
	// Sanity check the shape.
	if (shape.length === 0) {
		shape = [1]; // a scalar
	}
	let num = 1;
	for (let i = 0; i < shape.length; i++) {
		num *= shape[i]!;
	}
	const fl = formatLength(format);
	if (bytes.length !== num * fl) {
		throw new IgnorableError(`The tensor "${name}" is the wrong size ${bytes.length} bytes for its shape ${JSON.stringify(shape)} and format "${format}": should be ${num * fl} bytes.`);
	}
}

// Get the byte length of a given format constant. This may be a fractional
// value if the format is something like 4-bit data.
export function formatLength(format: Format): Integer {
	const num = format.match(/\d+/);
	if (!num) {
		throw new Error(`The element format "${format}" is not a valid format (must contain bit size).`);
	}
	const parsed = parseInt(num[0], 10) | 0;
	// This is an attempt to support sub-byte precision. It will work
	// for many cases, but doesn't deal with rounding-up of un-aligned sizes.
	if ((parsed | 0) & ((parsed | 0) - 1)) {
		throw new Error(`The element format "${format}" is not a valid format (must be power of 2).`);
	}
	return parsed/8.0;
}
