package Loss;

public class MSE extends Loss {
    public double forward(double[] expected, double[] actual) {
        double sum = 0;
        for (int i = 0; i < expected.length; i++) {
            sum += Math.pow(expected[i]-actual[i], 2);
        }

        return 2 * sum / expected.length;
    }


    public double[] back(double[] expected, double[] actual) {
        double[] out = new double[expected.length];

        for (int i = 0; i < expected.length; i++) {
            out[i] = (2 / expected.length) * (actual[i] - expected[i]);
        }

        return out;

    }
}
