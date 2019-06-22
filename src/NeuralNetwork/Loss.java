/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NeuralNetwork;

import Control.MathFx;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 *
 * @author Azhary Arliansyah
 */
public class Loss {
    
    public static double[] crossEntropy2(double[] actual, double[] predicted) {
        double[] losses = new double[actual.length];
        for (int i = 0; i < actual.length; i++) {
            losses[i] = (actual[i] - predicted[i]) / (double)actual.length;
        }
        return losses;
    }
    
    public static double crossEntropy(double[] actual, double[] predicted) {
        double loss = 0.0;
        for (int i = 0; i < actual.length; i++) {
            loss += ((actual[i] * Math.log(predicted[i])) + 
                    ((1 - actual[i]) * Math.log(1 - predicted[i])));
        }
        loss *= -(1.0 / (double)actual.length);
        return loss;
    }
    
    public static double error(double[] actual, double[] predicted) {
        double error = 0.0;
        int actualIndex = MathFx.maxIndex(actual);
//        for (int i = 0; i < actual.length; i++) {
//            error += Math.pow(predicted[i] - actual[i], 2);
//        }
        return - Math.log(predicted[actualIndex]) / (double)actual.length;
    }
    
    public static double crossEntropyDerivative(double actual, 
            double predicted) {
        double loss = (-1 * ((actual * (1 / predicted)) + 
                    (1 - actual) * (1 / (1 - predicted))));
        return loss;
    }
}
