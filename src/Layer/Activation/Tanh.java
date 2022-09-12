package Layer.Activation;

import Layer.ActivationLayer;

public class Tanh extends ActivationLayer {

    public Tanh() {
        //super( (a) -> Math.tanh(a), (a) -> 1 - (Math.pow(Math.tanh(a), 2)));
        super( (a) -> Math.tanh(a), (a) -> 1 - Math.pow(Math.tanh(a), 2));
    }



}
