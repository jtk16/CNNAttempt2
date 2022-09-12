package Layer;

public class Dense extends Layer {
    public double[] inputs;
    public double[] outputs;

    public double[] biases;

    public double[][] weights;

    public int inputLength;
    public int outputLength;

    public Dense(int inputLength, int outputLength) {
        this.inputLength = inputLength;
        this.outputLength = outputLength;


        this.inputs = new double[inputLength];
        this.outputs = new double[outputLength];

        this.weights = new double[outputLength][inputLength];
        this.biases = new double[outputLength];

        for (int i = 0; i < inputLength; i++) inputs[i] = Math.random() - 0.5;

        for (int i = 0; i < outputLength; i++) biases[i] = Math.random() - 0.5;

        for (int outIdx = 0; outIdx < outputLength; outIdx++) {
            for (int inIdx = 0; inIdx < inputs.length; inIdx++) {
                this.weights[outIdx][inIdx] = Math.random() - 0.5;
            }
        }
    }


    @Override
    public double[] feedForward(double[] fInput) {
        this.inputs = fInput;

        double[] out = new double[outputLength];

        for (int outIdx = 0; outIdx < outputLength; outIdx++) {
            double sum = 0;
            for (int inIdx = 0; inIdx < inputLength; inIdx++) {
                sum += weights[outIdx][inIdx] * inputs[inIdx];
            }
            out[outIdx] = sum + biases[outIdx]; //+biases[outIdx]
        }
        this.outputs = out;

        return out;
    }

    @Override
    public double[] backPropagate(double[] incomingGradient, double l_rate) {

        double[] previousLayerGradient = new double[inputLength];

        for (int i = 0; i < inputLength; i++) {
            double sum = 0;
            for (int j = 0; j < outputLength; j++) {
                sum += incomingGradient[j] * weights[j][i] * l_rate;
            }
            previousLayerGradient[i] = sum;
        }

        for (int i = 0; i < inputLength; i++) {
            for (int j = 0; j < outputLength; j++) {
                weights[j][i] -= incomingGradient[j] * inputs[i] * l_rate;
            }
        }

        for (int j = 0; j < outputLength; j++) {
            biases[j] -= incomingGradient[j] * l_rate;
        }

        return previousLayerGradient;

    }
}
