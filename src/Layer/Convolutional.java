package Layer;

import Util.ConvolutionalMath;
import Util.Size;

import java.lang.reflect.Array;

public class Convolutional extends Layer {
    public int inputDepth;
    public int inputWidth;
    public int inputHeight;
    public Size inputSize;

    public int kernelDimensions;

    public Size kernelLayerSize;
    public int kernelLayerDepth;

    public Size outputSize;

    public double[][][][] kernels;
    public double[][][] biases;

    public double[][][] inputs;
    public double[][][] outputs;

    public Convolutional(Size inputSize, int kernelSize, int depth) {
        this.inputDepth = inputSize.x;
        this.inputHeight = inputSize.y;
        this.inputWidth = inputSize.z;
        this.inputSize = inputSize;

        this.kernelDimensions = kernelSize;
        this.kernelLayerDepth = depth;
        this.kernelLayerSize = new Size(depth, inputDepth, kernelSize);

        this.outputSize = new Size(depth, inputHeight - kernelSize + 1, inputWidth - kernelSize + 1);

        this.kernels = new double[depth][inputDepth][kernelSize][kernelSize];
        this.biases = new double[depth][outputSize.y][outputSize.z];
        this.outputs = new double[depth][inputHeight - kernelSize + 1][inputWidth - kernelSize + 1];

        for (int i = 0; i < depth; i++) {
            for (int j = 0; j < inputDepth; j++) {
                for (int k = 0; k < kernelSize; k++) {
                    for (int l = 0; l < kernelSize; l++) {
                        this.kernels[i][j][k][l] = Math.random() - 0.5;
                    }
                }
            }

            for (int j = 0; j < outputSize.y; j++) {
                for (int k = 0; k < outputSize.z; k++) {
                    this.biases[i][j][k] = Math.random() - 0.5;
                }
            }
        }
    }

    @Override
    public double[][][] feedForward(double[][][] fInput) {
        this.inputs = fInput;
        this.outputs = biases;

        for (int d = 0; d < kernelLayerDepth; d++) {
            for (int j = 0; j < inputDepth; j++) {
                double[][] currentKernel = kernels[d][j];

                double[][] add = ConvolutionalMath.correlate2DValid(inputs[j], currentKernel);

                for (int x = 0; x < add.length; x++) {
                    for (int y = 0; y < add[0].length; y++) {
                        outputs[d][x][y] += add[x][y];
                    }
                }
            }
        }

        return this.outputs;
    }

    @Override
    public double[][][] backPropagate(double[][][] incomingGradient, double l_rate) {
        double[][][][] kernelGradient = new double[kernelLayerDepth][inputDepth][kernelDimensions][kernelDimensions];
        double[][][] inputGradient = new double[inputSize.x][inputSize.y][inputSize.z];

        for (int i = 0; i < kernelLayerDepth; i++) {
            for (int j = 0; j < inputDepth; j++) {
                kernelGradient[i][j] = ConvolutionalMath.correlate2DValid(inputs[j], incomingGradient[i]);

                double[][] add = ConvolutionalMath.convolve2DFull(incomingGradient[i], kernels[i][j]);

                for (int x = 0; x < add[0].length; x++) {
                    for (int y = 0; y < add.length; y++) {
                        inputGradient[j][y][x] += add[y][x];
                    }
                }
            }
        }

        for (int i = 0; i < kernelLayerDepth; i++) {
            for (int j = 0; j < inputDepth; j++) {
                for (int y = 0; y < kernelDimensions; y++) {
                    for (int x = 0; x < kernelDimensions; x++) {
                        kernels[i][j][y][x] -= l_rate * kernelGradient[i][j][y][x];
                    }
                }
            }
        }

        for (int d = 0; d < kernelLayerDepth; d++) {
            for (int j = 0; j < outputSize.y; j++) {
                for (int i = 0; i < outputSize.x; i++) {
                    biases[d][j][i] -= l_rate * incomingGradient[d][j][i];
                }
            }
        }

        return inputGradient;
    }
}
