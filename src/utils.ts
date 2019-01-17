import * as mkdirp from 'mkdirp';
import * as path from 'path';
import { promisify } from 'util';

const mkdir = promisify(mkdirp);
/**
 * Creates folders for the entire given path if necessary.
 * Same behaviour as mkdir -p
 */
export const createFolder = (location: string) => mkdir(path.dirname(location));

export function assertNever(t: never, msg = 'Unexpected `assertNever` branch reached') {
    return new Error(msg);
}

export type BufferType = Uint8Array | Float32Array | Int32Array;
