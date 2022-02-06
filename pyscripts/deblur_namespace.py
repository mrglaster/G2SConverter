class DeblurNamespace:
    def __init__(self, model, phase, datalist, batch_size, epoch, learning_rate, gpu, height, width, input_path, output_path):
        self.model= model
        self.phase = phase
        self.datalist = datalist
        self.batch_size = batch_size
        self.epoch = epoch
        self.learning_rate = learning_rate
        self.gpu = gpu
        self.height = height
        self.width = width
        self.input_path = input_path
        self.output_path = output_path






