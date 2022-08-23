import io
import onnx
import onnxruntime as ort
import copy

class ONNXModel(object):
    def __init__(self, onnx_path, config, is_check=True) -> None:
        self.onnx_path = onnx_path
        onnx_model = onnx.load(onnx_path)
        
        if is_check:
            onnx.checker.check_model(onnx_model)
            
    def _printable_onnx_graph(self):
        print(onnx.helper.printable_graph(self.onnx_model))
        
    def _get_inputs(self):
        return self.onnx_model.get_inputs()
    
    def _get_outputs(self):
        return self.onnx_model.get_outputs()
    
    def create_ort_session(self, config):
        ort_session = ort.InferenceSession(self.onnx_path, **config)
        ort_session.disable_fallback()
        
        self.ort_session = ort_session
        return self.ort_session
    
    def _io_binding(self, bind_config):
        ort_session = self.ort_session
        io_binding = ort_session.io_binding()
        
        if bind_config['inputs']:
            io_binding.bind_cpu_input(**bind_config['input'])
        
        if bind_config['outputs']:
            ort_session.clear_binding_outputs()
            io_binding.bind_output(**bind_config['outputs'])

        ort_session.run_with_iobinding(io_binding)
        
        return io_binding
    
    def predict(self, inp):
        pass