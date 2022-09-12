package Layer;

import java.util.function.Function;

public class ActivationLayer extends Layer {

    public Function<Double, Double> forward;
    public Function<Double, Double> back;

    public ActivationLayer(Function<Double, Double> forward, Function<Double, Double> back) {
        this.forward = forward;
        this.back = back;
    }

    @Override
    public double[] feedForward(double[] fInput) {
        double[] out = new double[fInput.length];
        this.inputs = fInput;

        for (int i = 0; i < out.length; i++) {
            out[i] = forward.apply(fInput[i]);
        }

        this.outputs = out;

        return out;
    }

    @Override
    public double[] backPropagate(double[] incomingGradient, double l_rate) {
        double[] previousLayerGradient = new double[incomingGradient.length];

        for (int i = 0; i < previousLayerGradient.length; i++) {

            previousLayerGradient[i] = incomingGradient[i] * back.apply(this.inputs[i]);

        }

        return previousLayerGradient;
    };

}
