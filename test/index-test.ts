import { parseSafeTensors, saveSafeTensors, TensorMap, TensorRef } from "@reve-ai/safetensors-parser";

const bytes = new Uint8Array([
    // header length: 64
    64, 0, 0, 0, 0, 0, 0, 0,
    // header (whitespace padded up to 64 bytes)
    0x7b, 0x22, 0x74, 0x65, 0x6e, 0x22, 0x3a, 0x7b, 0x22, 0x64, 0x74, 0x79,
    0x70, 0x65, 0x22, 0x3a, 0x22, 0x42, 0x46, 0x31, 0x36, 0x22, 0x2c, 0x22,
    0x73, 0x68, 0x61, 0x70, 0x65, 0x22, 0x3a, 0x5b, 0x32, 0x2c, 0x33, 0x5d,
    0x2c, 0x22, 0x64, 0x61, 0x74, 0x61, 0x5f, 0x6f, 0x66, 0x66, 0x73, 0x65,
    0x74, 0x73, 0x22, 0x3a, 0x5b, 0x30, 0x2c, 0x31, 0x32, 0x5d, 0x7d, 0x7d,
    0x20, 0x20, 0x20, 0x20,
    // tensor data
    0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 
    0x20, 0x20, 0x20, 0x20,
]);

const stuff = parseSafeTensors(bytes);
const ten = stuff.getTensor("ten");
if (!ten) {
    throw new Error("No tensor named ten");
}
if (ten.format !== "BF16") {
    throw new Error(`The tensor format is not BF16: ${ten.format}`);
}
if (ten.shape.length !== 2) {
    throw new Error(`The tensor shape is not length 2: ${ten.shape.length}`);
}
if (ten.shape[0] !== 2) {
    throw new Error(`The tensor shape is not [2, 3]: ${JSON.stringify(ten.shape)}`);
}
if (ten.shape[1] !== 3) {
    throw new Error(`The tensor shape is not [2, 3]: ${JSON.stringify(ten.shape)}`);
}
if (ten.bytes.length !== 12) {
    throw new Error(`The tensor bytes is not length 12: ${ten.bytes.length}`);
}
if (JSON.stringify(Array.from(ten.bytes)) !== JSON.stringify([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])) {
    throw new Error(`The tensor bytes is not [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]: ${JSON.stringify(Array.from(ten.bytes))}`);
}

const parsed = saveSafeTensors(stuff);

const a1 = Array.from(bytes);
const a2 = Array.from(parsed);

if (a1.length !== a2.length) {
    throw new Error(`The lengths are different: ${a1.length} !== ${a2.length}`);
}
for (let i = 0; i < a1.length; i++) {
    if (a1[i] !== a2[i]) {
        // This check is a little brittle, because there's no guarantee
        // that the JSON.stringify() will produce the same output as we
        // read in the JSON.parse().
        throw new Error(`Byte ${i} is different: ${a1[i]} !== ${a2[i]}`);
    }
}

const tm1 = new TensorMap();
tm1.addTensor("snark", new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]), "UINT8", [1, 3, 4]);
tm1.addTensor("snork", new Uint8Array([0, 1, 1, 0]), "UINT8", [4]);
tm1.addTensor("snurk", new TensorRef("snurk", new Uint8Array([0, 1, 1, 0]), "UINT8", [2, 2]));
tm1.setMetaValue("foo", "bar");
tm1.allMetadata.set("baz", "quux");

const tm2 = parseSafeTensors(saveSafeTensors(tm1));

if (tm1.allTensors.size !== tm2.allTensors.size) {
    throw new Error(`The tensor maps have different sizes ${tm1.allTensors.size} !== ${tm2.allTensors.size}`);
}
const tm1td = Array.from(tm1.allTensors.entries()).sort();
const tm2td = Array.from(tm2.allTensors.entries()).sort();
if (tm1td.length !== tm2td.length) {
    throw new Error(`The tensor maps have different lengths ${tm1td.length} !== ${tm2td.length}`);
}
for (let i = 0; i < tm1td.length; i++) {
    if (tm1td[i]![0]! !== tm2td[i]![0]!) {
        throw new Error(`The tensor maps have different keys: ${tm1td[i]![0]!} !== ${tm2td[i]![0]!}`);
    }
    if (tm1td[i]![1]!.name !== tm2td[i]![1]!.name) {
        throw new Error(`The tensor maps have different names: ${tm1td[i]![1]!.name} !== ${tm2td[i]![1]!.name}`);
    }
    if (tm1td[i]![1]!.format !== tm2td[i]![1]!.format) {
        throw new Error(`The tensor maps have different formats: ${tm1td[i]![1]!.format} !== ${tm2td[i]![1]!.format}`);
    }
    if (JSON.stringify(tm1td[i]![1]!.shape) !== JSON.stringify(tm2td[i]![1]!.shape)) {
        throw new Error(`The tensor maps have different shapes: ${JSON.stringify(tm1td[i]![1]!.shape)} !== ${JSON.stringify(tm2td[i]![1]!.shape)}`);
    }
    if (JSON.stringify(Array.from(tm1td[i]![1]!.bytes)) !== JSON.stringify(Array.from(tm2td[i]![1]!.bytes))) {
        throw new Error(`The tensor maps have different bytes: ${JSON.stringify(Array.from(tm1td[i]![1]!.bytes))} !== ${JSON.stringify(Array.from(tm2td[i]![1]!.bytes))}`);
    }
}

if (tm1.allMetadata.size !== tm2.allMetadata.size) {
    throw new Error(`The metadata maps have different sizes ${tm1.allMetadata.size} !== ${tm2.allMetadata.size}`);
}
const tm1md = Array.from(tm1.allMetadata.entries()).sort();
const tm2md = Array.from(tm2.allMetadata.entries()).sort();
if (tm1md.length !== tm2md.length) {
    throw new Error(`The metadata maps have different lengths ${tm1md.length} !== ${tm2md.length}`);
}
for (let i = 0; i < tm1md.length; i++) {
    if (tm1md[i]![0]! !== tm2md[i]![0]!) {
        throw new Error(`The metadata maps have different keys: ${tm1md[i]![0]!} !== ${tm2md[i]![0]!}`);
    }
    if (tm1md[i]![1]! !== tm2md[i]![1]!) {
        throw new Error(`The metadata maps have different values: ${tm1md[i]![1]!} !== ${tm2md[i]![1]!}`);
    }
}

console.log("All tests passed.");
