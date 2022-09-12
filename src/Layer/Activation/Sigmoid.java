package Layer.Activation;

import Layer.ActivationLayer;
import Testing.Activation;

import java.util.function.Function;

public class Sigmoid extends ActivationLayer {
    Function<Double, Double> forward = (a) -> 1.0/(1+Math.exp(-a));
    Function<Double, Double> back = (a) -> forward.apply(a) * (1.0-forward.apply(a));

    public Sigmoid() {
        super(
                (a) -> 1.0/(1+Math.exp(-a)),
                (a) -> (1.0/(1+Math.exp(-a))) * (1.0 - (1.0/(1+Math.exp(-a))))
        );
    }


}
