package Loss;

public abstract class Loss {


    public abstract double forward(double[] expected, double[] actual);

    public abstract double[] back(double[] expected, double[] actual);


}
