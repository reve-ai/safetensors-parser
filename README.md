# Safetensors Parser for Javascript

"safetensors" is the highest-performance file format in wide use within the
pytorch machine learning community. It's a very simple format, and the Python
libraries for dealing with the format are generally of good quality. However,
sometimes you need to send tensors around some web infrastructure that might
include a browser, or a node or deno server, and then what do you do?

This library includes a utility class to build and save safetensor files, as
well as parse, load, and inspect them. It validates the files and throws an
error when the file is somehow not up to snuff. Some of those validations can
be turned off if you like not knowing things.

Safetensors-parser attempts to be memory efficient, inasmuchas anything in this
area can be. Each separate tensor from a parsed file references the underlying
byte array. Each tensor is written as a separate chunk if you provide a write
callback to saving tensors. This could allow you to stream tensor data over a
network without having to keep all of them in RAM at once.

## Usage: Loading

```typescript
import { parseSafeTensors } from "@reve-ai/safetensors-parser";

const stuff: UInt8array = ...;

const tensorMap = parseSafeTensors(stuff);
const myTensor = tensorMap.getTensor("my-tensor");
```

## Usage: Saving

```typescript
import { saveSafeTensors, TensorMap } from "@reve-ai/safetensors-parser";

const tensorMap = new TensorMap();
tensorMap.setMetadata("creator", "me");
tensorMap.addTensor(
  "identity",
  new UInt8array([1, 0, 0, 0, 1, 0, 0, 0, 1], "UINT8", [3, 3])
);

// Use the default writer, which returns the full byte array.
// If you use a custom write callback, nothing will be returned.
const stuff: UInt8array = saveSafeTensors(tensorMap);
```

# FAQ

These questions might have been asked by some person at some point, making them
more frequently asked than questions that nobody has asked.

## Why does the package.json have no bundler?

This package is a proper module, provided in a single file (dist/src/index.js)
which you can just import as-is, no bundler required.

## Why does the package.json have no test runner?

The tests live in a single file that you can just run with node.js.

## Can you add convnient dependency wrappers for all my favorite frameworks and image loading libraries?

This package has no runtime dependencies, and the only development time
dependency is typescript, and it will stay that way.

## I like the Buffer class better than the UInt8array class.

That's not a question. Also, `Buffer` is not available in the browser; this
library is intended to be usable both in a browser and on a server.

# Detailed Usage

Other than `parseSafeTensors` and `saveSafeTensors`, the rest of the functions
are largely internal but are exported in case you want to use them or test them.
The `TensorMap` class is intended for direct usage, and you could also use
`TensorRef` directly.

## TensorMap

```typescript
class TensorMap {
  constructor();
  getTensor(name: TensorName): TensorRef | undefined;
  addTensor(
    name: TensorName,
    bytes: Bytes,
    format: Format,
    shape: Shape
  ): TensorRef;
  addTensor(name: TensorName, tensor: TensorRef): TensorRef;
  setTensor(name: TensorName, tensor: TensorRef): TensorRef;
  getOrMakeTensor(name: TensorName, factory: () => TensorRef): TensorRef;
  getMetaValue(name: TensorName): string | undefined;
  setMetaValue(name: TensorName, value: string): void;
  get allMetadata(): Map<string, string>;
  get allTensors(): Map<string, TensorRef>;
  setAllMetadata(metadata: Map<string, string>): void;
  removeTensor(name: TensorName | TensorRef): void;
}
```

`TensorMap` is a collection of multiple tensors. It can be manually constructed
or loaded from a safetensors file. It provides methods to manage tensors and
metadata.

- `constructor()`: Creates a new empty TensorMap.
- `getTensor(name)`: Retrieves a tensor by name, or returns undefined if not found.
- `addTensor(name, ...)`: Adds a new tensor to the map. Throws an error if the name already exists.
- `setTensor(name, tensor)`: Sets a tensor, replacing any existing tensor with the same name.
- `setMetaValue(name, value)`: Sets a metadata value.
- `allMetadata`: Getter that returns all metadata as a Map.
- `allTensors`: Getter that returns all tensors as a Map.
- `setAllMetadata(metadata)`: Sets all metadata at once.
- `removeTensor(name)`: Removes a tensor from the map.

## TensorRef

```typescript
class TensorRef {
  constructor(name: TensorName, bytes: Bytes, format: Format, shape: Shape);
  get parent(): TensorMap | undefined;
  get name(): string;
  set name(val: string);
  removeIfParented(): void;
  sanityCheck(): void;
}
```

`TensorRef` represents a specific tensor within a safetensors archive. It can be
created standalone or obtained from a TensorMap.

- `constructor(name, bytes, format, shape)`: Creates a new TensorRef with the given properties.
- `parent`: Getter that returns the parent TensorMap, if any.
- `name`: Getter and setter for the tensor name. Setting the name will update it in the parent TensorMap if present.
- `removeIfParented()`: Removes the tensor from its parent TensorMap, if it has one.
- `sanityCheck()`: Performs sanity checks on the tensor, ensuring its size matches the declared shape and format.

Note: The `bytes`, `format`, and `shape` properties are read-only and can be accessed directly on the TensorRef instance.

## parseSafeTensors

```typescript
function parseSafeTensors(
  bytes: Uint8Array,
  ignoreInvalid?: boolean
): TensorMap;
```

Parses a safetensors file (as a Uint8Array) and returns a TensorMap containing
the tensors and metadata stored in that file. If the file is not fully compliant
with the spec, an error will be thrown. You can pass `ignoreInvalid=true` to
attempt parsing anyway.

## saveSafeTensors

```typescript
function saveSafeTensors(tensorMap: TensorMap): Uint8Array;
function saveSafeTensors(
  tensorMap: TensorMap,
  write: (data: Uint8Array) => void
): undefined;
```

Generates the contents of a safetensors file from a given TensorMap. It can
either return a Uint8Array containing the file contents or call a provided write
function with chunks of data.

## sanityCheckTensorsHeader

```typescript
function sanityCheckTensorsHeader(
  ignoreInvalid: boolean,
  bytes: Uint8Array,
  filesize: number
): number;
```

Verifies that the header of a safetensors file seems legitimate. Returns the
size of the JSON chunk that starts at offset 8.

## unsafeGetHeaderSize

```typescript
function unsafeGetHeaderSize(bytes: Uint8Array): number;
```

Retrieves the header size from the first 4 bytes of a safetensors file. This
function is unsafe as it doesn't perform any checks.

## sanityCheckTensorsParsed

```typescript
function sanityCheckTensorsParsed(
  ignoreInvalid: boolean,
  j: Object,
  chunksize: number
): void;
```

Performs sanity checks on the parsed JSON content of a safetensors file.

## sanityCheck

```typescript
function sanityCheck(tensor: TensorRef): void;
function sanityCheck(bytes: Uint8Array, format: string, shape: number[]): void;
```

Performs sanity checks on a tensor, ensuring that its size matches its declared
shape and format.

## formatLength

```typescript
function formatLength(format: string): number;
```

Returns the byte length of a given tensor format. This may be a fractional value
for formats like 4-bit data.
