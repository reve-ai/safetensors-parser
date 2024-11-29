"use strict";
Object.defineProperty(exports, "__esModule", { value: true });
exports.TensorRef = exports.TensorMap = exports.IgnorableError = void 0;
exports.parseSafeTensors = parseSafeTensors;
exports.saveSafeTensors = saveSafeTensors;
exports.sanityCheckTensorsHeader = sanityCheckTensorsHeader;
exports.unsafeGetHeaderSize = unsafeGetHeaderSize;
exports.sanityCheckTensorsParsed = sanityCheckTensorsParsed;
exports.sanityCheck = sanityCheck;
exports.formatLength = formatLength;
class IgnorableError extends Error {
    constructor(message) {
        super(message + " You can ignore this error with ignoreInvalid=true in parseSafeTensors().");
        this.name = "IgnorableError";
    }
}
exports.IgnorableError = IgnorableError;
// Given a safetensors file (as a Uint8Array) return the TensorMap of the
// tensors and metadata stored in that file. If the file is not fully compliant
// with the spec (additional properties, etc) an error will be thrown. If you'd
// like to try using it anyway, you can pass ignoreInvalid=true.
function parseSafeTensors(bytes, ignoreInvalid) {
    sanityCheckTensorsHeader(!!ignoreInvalid, bytes, bytes.length);
    const hdrsize = unsafeGetHeaderSize(bytes);
    const ret = new TensorMap();
    const j = JSON.parse(new TextDecoder().decode(bytes.slice(8, 8 + hdrsize)));
    sanityCheckTensorsParsed(!!ignoreInvalid, j, bytes.length - 8 - hdrsize);
    for (const [name, val] of Object.entries(j)) {
        if (name === "__metadata__") {
            ret.setAllMetadata(new Map(Object.entries(val)));
        }
        else {
            const { dtype, shape, data_offsets } = val;
            ret.addTensor(name, bytes.slice(8 + hdrsize + data_offsets[0], 8 + hdrsize + data_offsets[1]), dtype, shape);
        }
    }
    return ret;
}
function saveSafeTensors(tensorMap, write) {
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
    tensorMap;
    write;
    hdr = {};
    constructor(tensorMap, write) {
        this.tensorMap = tensorMap;
        this.write = write;
        this.setMetadata();
    }
    tes;
    offset;
    hdrBuf;
    lenbuf;
    hblen;
    ret;
    setMetadata() {
        const md = {};
        let toset = false;
        for (const [name, val] of this.tensorMap.allMetadata.entries()) {
            md[name] = val;
            toset = true;
        }
        if (toset) {
            this.hdr["__metadata__"] = md;
        }
    }
    calcOffset() {
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
    calcHeader() {
        const hdrBuf = (new TextEncoder()).encode(JSON.stringify(this.hdr));
        const hblen = (hdrBuf.length + 7) & ~7;
        if (hblen > 100 * 1024 * 1024) {
            throw new Error("The metadata is too large to be saved in a safetensors file.");
        }
        this.lenbuf = new Uint8Array([hblen & 0xff, (hblen >> 8) & 0xff, (hblen >> 16) & 0xff, (hblen >> 24) & 0xff, 0, 0, 0, 0]);
        this.hdrBuf = hdrBuf;
        this.hblen = hblen;
        return hblen;
    }
    writeHeader() {
        if (!this.write) {
            this.ret = new Uint8Array(this.offset + this.hblen + 8);
            let wrote = 0;
            this.write = (data) => {
                this.ret.set(data, wrote);
                wrote += data.length;
            };
        }
        this.write(this.lenbuf);
        this.write(this.hdrBuf);
        if (this.hblen > this.hdrBuf.length) {
            this.write(padder.slice(0, this.hblen - this.hdrBuf.length));
        }
    }
    writeTensors() {
        let written = 0;
        for (const [name, tensor] of this.tes) {
            this.write(tensor.bytes);
            written += tensor.bytes.length;
            if (tensor.bytes.length & 7) {
                this.write(padder.slice(0, 8 - (tensor.bytes.length & 7)));
                written += 8 - (tensor.bytes.length & 7);
            }
            const end = this.hdr[name]["data_offsets"][1];
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
class TensorMap {
    refs = new Map();
    metadata = new Map();
    constructor() { }
    // If the tensor exists in the map, return it, else return undefined.
    getTensor(name) {
        return this.refs.get(name);
    }
    addTensor(name, data, format, shape) {
        if (this.refs.get(name)) {
            throw new Error(`The name "${name}" already exists in the TensorMap.`);
        }
        if (data instanceof TensorRef) {
            return this._setTensor(name, data);
        }
        if (!format || !shape) {
            throw new Error(`You must provide format and shape when adding tensor "${name}".`);
        }
        else {
            return this._setTensor(name, new TensorRef(name, data, format, shape));
        }
    }
    // Set this name to be this tensor, no matter whether it exists or not. Remove
    // any previous tensor of the same name (which may be the same value!)
    setTensor(name, tensor) {
        this.removeTensor(name);
        return this.addTensor(name, tensor);
    }
    // If the tensor exists, then return it. Otherwise, create it with the factory
    // function, add it to the map, and return it.
    getOrMakeTensor(name, factory) {
        let ten = this.refs.get(name);
        if (ten) {
            return ten;
        }
        ten = factory();
        this.refs.set(name, ten);
        return ten;
    }
    // If some metadata value exists, return it, else return undefined.
    getMetaValue(name) {
        return this.metadata.get(name);
    }
    setMetaValue(name, value) {
        this.metadata.set(name, value);
    }
    get allMetadata() {
        return this.metadata;
    }
    get allTensors() {
        return this.refs;
    }
    setAllMetadata(metadata) {
        this.metadata = metadata;
    }
    // Remove a tensor from the map. It's OK if it doesn't exist.
    removeTensor(name) {
        let ten;
        if (typeof name === "string") {
            ten = this.refs.get(name);
            if (!ten) {
                return;
            }
        }
        else {
            ten = name;
        }
        if (ten.parent !== this) {
            throw new Error("You can only remove a tensor from the TensorMap that it belongs to.");
        }
        delete ten._parent;
        this.refs.delete(ten.name);
    }
    _setTensor(name, tensor) {
        if (tensor.parent && tensor.parent !== this) {
            throw new Error("You can only add a tensor to a TensorMap that it doesn't already belong to.");
        }
        this.refs.set(name, tensor);
        tensor._parent = this;
        return tensor;
    }
}
exports.TensorMap = TensorMap;
;
// TensorRef is a specific tensor within a safetensors archive. You can make one free
// standing, or get one from an archive.
class TensorRef {
    bytes;
    format;
    shape;
    // Make a new tensorref referencing the given data. The new tensorref is not parented.
    // Note that the shape and format are not sanity checked! Hopefully the bytes are accurate.
    constructor(name, bytes, format, shape) {
        this.bytes = bytes;
        this.format = format;
        this.shape = shape;
        this._name = name;
    }
    // A tensorref may have a parent tensor map, or may be freestanding.
    get parent() { return this._parent; }
    _name;
    _parent;
    // If you mutate the name of a tensor in a tensor map, it will rename the tensor
    // in the map. It's an error to rename to a name that already exists in the map.
    get name() { return this._name; }
    set name(val) {
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
}
exports.TensorRef = TensorRef;
;
// Given the bytes of a safetensors file, verify that the header seems legitimate.
// Note that you only need 10 bytes to check this. The value returned is the size
// of the JSON chunk that starts at offset 8, so if the total file size is smaller
// than 8+return, the file is likely truncated.
function sanityCheckTensorsHeader(ignoreInvalid, bytes, filesize) {
    if (!arrayCompare(Array.from(bytes.slice(4, 9)), [0, 0, 0, 0, "{".charCodeAt(0)])) {
        throw new Error("The file header is not a valid safetensors file.");
    }
    const val = unsafeGetHeaderSize(bytes);
    if (val > 100 * 1024 * 1024 && !ignoreInvalid) {
        throw new IgnorableError("The file header is too long to be a valid safetensors file.");
    }
    if (filesize < 8 + val) {
        throw new Error("The safetensors file seems truncated or otherwise not valid.");
    }
    return val;
}
function unsafeGetHeaderSize(bytes) {
    return ((bytes[0] | 0) + ((bytes[1] | 0) * 256) + ((bytes[2] | 0) * 256 * 256) + ((bytes[3] | 0) * 256 * 256 * 256)) | 0;
}
// Return true if each element compares equal in the arrays
function arrayCompare(a, b) {
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
function sanityCheckTensorsParsed(ignoreInvalid, j, chunksize) {
    const covered = [];
    for (const [name, value] of Object.entries(j)) {
        if (name === "__metadata__") {
            checkMetadata(ignoreInvalid, value);
            continue;
        }
        const tac = new TensorAttributeChecker(chunksize, covered, ignoreInvalid);
        tac.checkTensorValue(name, value);
    }
}
function checkMetadata(ignoreInvalid, value) {
    for (const [key, strval] of Object.entries(value)) {
        if (typeof strval !== "string" && !ignoreInvalid) {
            throw new IgnorableError(`The metadata value ${key} is not a string (${typeof strval}).`);
        }
    }
}
class TensorAttributeChecker {
    chunksize;
    covered;
    ignoreInvalid;
    has_dtype;
    has_shape;
    has_data_offsets;
    constructor(chunksize, covered, ignoreInvalid = false, has_dtype = false, has_shape = false, has_data_offsets = false) {
        this.chunksize = chunksize;
        this.covered = covered;
        this.ignoreInvalid = ignoreInvalid;
        this.has_dtype = has_dtype;
        this.has_shape = has_shape;
        this.has_data_offsets = has_data_offsets;
    }
    dtype(key, val) {
        if (typeof val !== "string") {
            throw new Error(`The dtype for ${key} is not a string (${typeof val}).`);
        }
        this.has_dtype = true;
    }
    shape(key, val) {
        this.checkTensorShape(key, val);
        this.has_shape = true;
    }
    data_offsets(key, val) {
        this.checkDataOffsets(key, val);
        this.has_data_offsets = true;
    }
    attrOk(key) {
        return key === "dtype" || key === "shape" || key === "data_offsets";
    }
    assertAttributePresence(name) {
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
    checkTensorShape(key, val) {
        if (!Array.isArray(val)) {
            throw new Error(`The shape for ${key} is not an array (${typeof val})`);
        }
        for (const v of val) {
            if (typeof v !== "number" && !this.ignoreInvalid) {
                throw new IgnorableError(`The shape for ${key} is not an array of numbers (${typeof v})`);
            }
        }
    }
    checkDataOffsets(key, obj) {
        if (!Array.isArray(obj) || obj.length !== 2) {
            throw new Error(`The data_offsets for ${key} is not an array with length 2 (${typeof obj})`);
        }
        this.checkDataOffsetBasics(key, obj);
        for (const [start, end] of this.covered) {
            this.checkTensorOverlap(key, start, end, obj);
        }
        this.covered.push(obj);
    }
    checkDataOffsetBasics(key, val) {
        let err = null;
        if (!pairInRange(val, this.chunksize) && !this.ignoreInvalid) {
            err = new IgnorableError(`The data_offsets for ${key} is out of range (${val[0]}-${val[1]} versus ${this.chunksize}).`);
        }
        if (err && !this.ignoreInvalid) {
            throw err;
        }
    }
    checkTensorOverlap(key, start, end, val) {
        let err = null;
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
    checkTensorValue(name, value) {
        for (const [key, val] of Object.entries(value)) {
            if (!this.attrOk(key)) {
                if (!this.ignoreInvalid) {
                    throw new IgnorableError(`The tensor description for "${name}" has an invalid key "${key}".`);
                }
                continue;
            }
            this[key](name, val);
        }
        this.assertAttributePresence(name);
    }
}
function pairInRange(pair, chunksize) {
    return pair[0] >= 0 && pair[0] <= chunksize && pair[1] >= 0 && pair[1] <= chunksize && pair[0] <= pair[1];
}
function sanityCheck(bytes, format, shape) {
    if (bytes instanceof TensorRef) {
        _sanityCheck(bytes.name, bytes.bytes, bytes.format, bytes.shape);
    }
    else {
        _sanityCheck("created", bytes, format, shape);
    }
}
function _sanityCheck(name, bytes, format, shape) {
    if (!format || !shape) {
        throw new Error(`The tensor "${name}" is missing format and shape information.`);
    }
    // Sanity check the shape.
    if (shape.length === 0) {
        shape = [1]; // a scalar
    }
    let num = 1;
    for (let i = 0; i < shape.length; i++) {
        num *= shape[i];
    }
    const fl = formatLength(format);
    if (bytes.length !== num * fl) {
        throw new IgnorableError(`The tensor "${name}" is the wrong size ${bytes.length} bytes for its shape ${JSON.stringify(shape)} and format "${format}": should be ${num * fl} bytes.`);
    }
}
// Get the byte length of a given format constant. This may be a fractional
// value if the format is something like 4-bit data.
function formatLength(format) {
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
    return parsed / 8.0;
}
//# sourceMappingURL=index.js.map