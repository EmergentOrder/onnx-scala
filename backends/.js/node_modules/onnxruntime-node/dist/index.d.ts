declare module 'onnxruntime-node' {
	export * from 'onnxruntime-common';
	export { listSupportedBackends } from './backend';

import { Backend, InferenceSession, InferenceSessionHandler } from 'onnxruntime-common';
import { Binding } from './binding';
declare class OnnxruntimeBackend implements Backend {
    init(): Promise<void>;
    createInferenceSessionHandler(pathOrBuffer: string | Uint8Array, options?: InferenceSession.SessionOptions): Promise<InferenceSessionHandler>;
}
export declare const onnxruntimeBackend: OnnxruntimeBackend;
export declare const listSupportedBackends: () => Binding.SupportedBackend[];
export {};

import { InferenceSession, OnnxValue } from 'onnxruntime-common';
type SessionOptions = InferenceSession.SessionOptions;
type FeedsType = {
    [name: string]: OnnxValue;
};
type FetchesType = {
    [name: string]: OnnxValue | null;
};
type ReturnType = {
    [name: string]: OnnxValue;
};
type RunOptions = InferenceSession.RunOptions;
/**
 * Binding exports a simple synchronized inference session object wrap.
 */
export declare namespace Binding {
    interface InferenceSession {
        loadModel(modelPath: string, options: SessionOptions): void;
        loadModel(buffer: ArrayBuffer, byteOffset: number, byteLength: number, options: SessionOptions): void;
        readonly inputNames: string[];
        readonly outputNames: string[];
        run(feeds: FeedsType, fetches: FetchesType, options: RunOptions): ReturnType;
        dispose(): void;
    }
    interface InferenceSessionConstructor {
        new (): InferenceSession;
    }
    interface SupportedBackend {
        name: string;
        bundled: boolean;
    }
}
export declare const binding: {
    InferenceSession: Binding.InferenceSessionConstructor;
    listSupportedBackends: () => Binding.SupportedBackend[];
};
export {};
export declare const version = '1.19.2';
}
