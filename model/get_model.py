from model import spike2flow

def get_model(args):
    model = spike2flow.Spike2Flow(args)
    return model