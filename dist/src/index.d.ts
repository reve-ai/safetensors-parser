export type Bytes = Uint8Array;
export type Format = string;
export type Integer = number;
export type OffsetPair = [Integer, Integer];
export type Shape = Integer[];
export type TensorName = string;
export declare class IgnorableError extends Error {
    constructor(message: string);
}
export declare function parseSafeTensors(bytes: Bytes, ignoreInvalid?: boolean): TensorMap;
export declare function saveSafeTensors(tensorMap: TensorMap): Uint8Array;
export declare function saveSafeTensors(tensorMap: TensorMap, write: (data: Bytes) => void): undefined;
export declare class TensorMap {
    refs: Map<string, TensorRef>;
    metadata: Map<string, string>;
    constructor();
    getTensor(name: TensorName): TensorRef | undefined;
    addTensor(name: TensorName, bytes: Bytes, format: Format, shape: Shape): TensorRef;
    addTensor(name: TensorName, tensor: TensorRef): TensorRef;
    setTensor(name: TensorName, tensor: TensorRef): TensorRef;
    getOrMakeTensor(name: TensorName, factory: () => TensorRef): TensorRef;
    getMetaValue(name: TensorName): string | undefined;
    setMetaValue(name: TensorName, value: string): void;
    get allMetadata(): Map<string, string>;
    get allTensors(): Map<string, TensorRef>;
    setAllMetadata(metadata: Map<string, string>): void;
    removeTensor(name: TensorName | TensorRef): void;
    private _setTensor;
}
export declare class TensorRef {
    readonly bytes: Bytes;
    readonly format: Format;
    readonly shape: Shape;
    constructor(name: TensorName, bytes: Bytes, format: Format, shape: Shape);
    get parent(): TensorMap | undefined;
    private _name;
    _parent?: TensorMap;
    get name(): string;
    set name(val: string);
    removeIfParented(): void;
    sanityCheck(): void;
}
export declare function sanityCheckTensorsHeader(ignoreInvalid: boolean, bytes: Bytes, filesize: Integer): Integer;
export declare function unsafeGetHeaderSize(bytes: Bytes): Integer;
export declare function sanityCheckTensorsParsed(ignoreInvalid: boolean, j: Object, chunksize: Integer): void;
export declare function sanityCheck(tensor: TensorRef): void;
export declare function sanityCheck(bytes: Bytes, format: Format, shape: Shape): void;
export declare function formatLength(format: Format): Integer;
//# sourceMappingURL=index.d.ts.map