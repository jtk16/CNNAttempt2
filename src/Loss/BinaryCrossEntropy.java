package Loss;

public class BinaryCrossEntropy extends Loss {

    @Override
    public double forward(double[] expected, double[] actual) {
        double n = expected.length;

        double sum = 0;

        for (int i = 0; i < expected.length; i++) {
            sum += expected[i]*Math.log(actual[i]) + (1.0-expected[i])*Math.log(1-actual[i]);
        }

        sum *= -1.0/n;

        return sum;
    }

    @Override
    public double[] back(double[] expected, double[] actual) {
        double[] gradient = new double[expected.length];

        for (int i = 0; i < expected.length; i++) {
            gradient[i] = (1.0/expected.length) * (((1.0 - expected[i])/(1.0 - actual[i])) - (expected[i] / actual[i]));
        }

        return gradient;
    }
}
