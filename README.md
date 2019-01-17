# idx-data

This is a tiny `node.js` module for reading/writing idx formatted binary data from/to the disk.
More information on the idx file format can be find [here](http://www.fon.hum.uva.nl/praat/manual/IDX_file_format.html) and [here](http://yann.lecun.com/exdb/mnist/).
The idx format is commonly used for sharing data in the machine learning community due to its smaller footprint than csv, its ability to represent tensor data, and its performance advantages while loading into memory.

## Format specification
The idx format is a binary file format with the following structure (where `[n]` represents the nth **byte**):
```
[0-1]: always 0
[2]: data type
[3]: number of dimensions
```
The data type in byte 2 is coded:
```
0x08: unsigned byte
0x09: signed byte
0x0B: short (2 bytes)
0x0C: int (4 bytes)
0x0D: float (4 bytes)
0x0E: double (8 bytes)
```
The remaining bytes in the header specify the size of each dimension using 4-byte integers.
For instance, if byte 3 indicated that there are 2 dimensions of data then we'd see:
```
[4-7]: size of dimension 1
[8-11]: size of dimension 2
```
All of the remaining bytes specify the data itself, where the size (in bytes) of each data point is specified by the value given by the 3rd byte in the header.

Although most commercial processors use little endian, this file format specifies the use of big endian byte representation.
This means all binary files will utilize big endian.

## Usage Example
```typescript
import * as idx from 'idx-data';

const [
    X_train,
    Y_train,
    X_test,
    Y_test,
] = await Promise.all([
    idx.loadBits('mnist/train-images-idx3-ubyte'),
    idx.loadBits('mnist/train-labels-idx1-ubyte'),
    idx.loadBits('mnist/t10k-images-idx3-ubyte'),
    idx.loadBits('mnist/t10k-labels-idx1-ubyte'),
]);

console.log(X_train); // =>
/*
{
    data: <Uint8Array>,
    shape: [60000, 28, 28],
    type: 'uint8',
}
*/
```
### With @tensorflow/tfjs
```typescript
// get X_train with above code
import * as tf from '@tensorflow/tfjs';

// Tensorflow will infer the type as uint8
// based on the type of the buffer passed in
const X = tf.tensor3d(X_train.data, X_train.shape);
X.print(true);
```

## API
### IdxTensor
Basic interface for returned data.

Note: the `type` field will always be redundant with the typed array used to store the data.
This field is provided to bypass `instanceof` checks in favor of string comparison checks.
```typescript
interface IdxTensor {
    data: Uint8Array | Float32Array | Int32Array;;
    shape: number[];
    type: 'float32' | 'int32' | 'uint8';;
}
```

### loadBits
Takes a `string` filepath and returns a `Promise<IdxTensor>`.
```typescript
const data = await idx.loadBits('path/to/file');
console.log(data); // =>
/*
{
    data: <Float32Array>,
    shape: [100, 10],
    type: 'float32',
}
*/
```

### saveBits
Takes a typed array (e.g. `Float32Array`) containing the data to be saved, a `number[]` containing the shape of the data, and a `string` filepath where the data will be saved.
Returns a `Promise<void>` that will be resolved once the file is written.

This method ensures that the data is written in big endian mode, even if this is not the native architecture of the host machine.
```typescript
// Note this will contain all 0
const data = new Float32Array(10);
const shape = [5, 2];

await idx.saveBits(data, shape, 'path/to/file.idx');
/*
Will write something like:
(Note: line breaks and comments not included in the files)
0x00 0x00               # always 0
0x0D 0x02               # data type: float32 with 2 dimensions
0x05 0x00 0x00 0x00     # first dimension of size 5
0x02 0x00 0x00 0x00     # second dimension of size 2

0x00 0x00 0x00 0x00     # all data takes 4-bytes to represent
0x00 0x00 0x00 0x00     # because it is float32
0x00 0x00 0x00 0x00     # each line represents a single data point
0x00 0x00 0x00 0x00
0x00 0x00 0x00 0x00
0x00 0x00 0x00 0x00
0x00 0x00 0x00 0x00
0x00 0x00 0x00 0x00
0x00 0x00 0x00 0x00
0x00 0x00 0x00 0x00
*/
```
