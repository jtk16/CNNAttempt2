package Layer;

import Util.ConvolutionalMath;
import Util.Size;

public class Reshape extends Layer {
    public Size inputShape;
    public Size outputShape;

    public Reshape(Size inputShape, Size outputShape) {
        this.inputShape = inputShape;
        this.outputShape = outputShape;
    }

    @Override
    public double[][][] feedForward(double[][][] fInput) {
        return ConvolutionalMath.reshape(fInput, outputShape);
    }

    @Override
    public double[][][] backPropagate(double[][][] incomingGradient, double l_rate) {
        return ConvolutionalMath.reshape(incomingGradient, inputShape);
    }
}
