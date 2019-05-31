/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NeuralNetwork;

import NeuralNetwork.Neuron.Type;
import Control.MathFx;
import java.io.FileWriter;
import java.io.IOException;
import java.util.Arrays;
import org.json.simple.JSONObject;

/**
 *
 * @author Azhary Arliansyah
 */
public class NeuralNetwork {
    
    private final double learningRate = 0.01;
    private final int EPOCH = 5;
    
    private final int numInputNeuron;
    private final int numHiddenNeuron1;
    private final int numHiddenNeuron2;
    private final int numOutputNeuron;
    
    private Neuron[] inputNeurons;
    private Neuron[] hiddenNeurons1;
    private Neuron[] hiddenNeurons2;
    private Neuron[] outputNeurons;
    
    private double[][] inputHidden1Connections;
    private double[][] hidden1Hidden2Connections;
    private double[][] hidden2OutputConnections;
    
    private double[] deltaInput;
    private double[] deltaHidden1;
    private double[] deltaHidden2;
    
    private double[] crossEntropyDerivatives;
    private double[][] data;
    private double[][] target;
    
    private double loss;
    
    public NeuralNetwork(double[][] data, double[][] target, 
            int numHiddens) {
        this.data = data;
        this.target = target;
        
        this.numInputNeuron = this.data[0].length;
        this.numHiddenNeuron1 = numHiddens;
        this.numHiddenNeuron2 = numHiddens;
        this.numOutputNeuron = this.target[0].length;
        
        this.initializeInputNeurons();
        this.initializeHiddenNeurons1();
        this.initializeHiddenNeurons2();
        this.initializeOutputNeurons();
    
        this.initializeInputHidden1Connections();
        this.initializeHidden1Hidden2Connections();
        this.initializeHidden2OutputConnections();
    }
    
    public void saveWeight() {
        JSONObject inputHidden1Weights = new JSONObject();
        for (int i = 0; i < this.inputHidden1Connections.length; i++) {
            JSONObject rows = new JSONObject();
            for (int j = 0; j < this.inputHidden1Connections[i].length; j++) {
                rows.put(j, this.inputHidden1Connections[i][j]);
            }
            inputHidden1Weights.put(i, rows);
        }
        
        JSONObject hidden1Hidden2Weights = new JSONObject();
        for (int i = 0; i < this.hidden1Hidden2Connections.length; i++) {
            JSONObject rows = new JSONObject();
            for (int j = 0; j < this.hidden1Hidden2Connections[i].length; j++) {
                rows.put(j, this.hidden1Hidden2Connections[i][j]);
            }
            hidden1Hidden2Weights.put(i, rows);
        }
        
        JSONObject hidden2OutputWeights = new JSONObject();
        for (int i = 0; i < this.hidden2OutputConnections.length; i++) {
            JSONObject rows = new JSONObject();
            for (int j = 0; j < this.hidden2OutputConnections[i].length; j++) {
                rows.put(j, this.hidden2OutputConnections[i][j]);
            }
            hidden2OutputWeights.put(i, rows);
        }
        
        try {
            FileWriter writer = new FileWriter("InputHidden1Weights.json");
            writer.write(inputHidden1Weights.toJSONString());
            writer.flush();
            
            writer = new FileWriter("Hidden1Hidden2Weights.json");
            writer.write(hidden1Hidden2Weights.toJSONString());
            writer.flush();
            
            writer = new FileWriter("Hidden2OutputWeights.json");
            writer.write(hidden2OutputWeights.toJSONString());
            writer.flush();
        }
        catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public void fit() {
        for (int e = 0; e < this.EPOCH; e++) {
            for (int i = 0; i < this.data.length; i++) {
                this.feedforward(this.data[i]);
                this.loss = this.calculateLoss(this.target[i]);
                this.backpropagation(this.target[i]);
            }
            
            for (double[] w : this.hidden1Hidden2Connections) {
                System.out.println(Arrays.toString(w));
            }
            System.out.println("EPOCH " + (e + 1) + " LOSS: " + this.loss);
        }
        
        this.saveWeight();
    }
    
    public void predict(double[][] data) {
        
    }
    
    private void feedforward(double[] data) {
        this.calculateInputHidden1(data);
        this.calculateHidden1Hidden2();
        this.calculateHidden2Output();
    }
    
    private void backpropagation(double[] actual) {
        this.calculateOutputHidden2(actual);
        this.calculateHidden2Hidden1();
        this.calculateHidden1Input();
    }
    
    private void calculateOutputHidden2(double[] actual) {
        this.crossEntropyDerivatives = new double[this.numOutputNeuron];
        double[] outputNeuronValues = new double[this.numOutputNeuron];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            this.crossEntropyDerivatives[i] = 
                    Loss.crossEntropyDerivative(actual[i], 
                            this.outputNeurons[i].getMappedValue());
            outputNeuronValues[i] = this.outputNeurons[i].getValue();
        }
        
        double[] softmaxDerivatives = 
                Activation.softmaxDerivative(outputNeuronValues);
        
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            this.deltaHidden2[i] = 0.0;
            for (int j = 0; j < this.numOutputNeuron; j++) {
                this.deltaHidden2[i] += this.crossEntropyDerivatives[j] * 
                        softmaxDerivatives[j] * 
                        this.hiddenNeurons2[i].getMappedValue();
                this.hidden2OutputConnections[j][i] += (this.learningRate * 
                        this.crossEntropyDerivatives[j] * 
                        softmaxDerivatives[j]);
            }
            
        }
        
//        for (int i = 0; i < this.numOutputNeuron; i++) {
//            for (int j = 0; j < this.numHiddenNeuron2; j++) {
//                this.deltaHidden2Output[i][j] = 
//                        this.crossEntropyDerivatives[i] * softmaxDerivatives[i] 
//                        * this.hiddenNeurons2[j].getMappedValue();
//                this.hidden2OutputConnections[i][j] += 
//                        (this.learningRate * this.deltaHidden2Output[i][j]);
//            }
//        }
    }
    
    private void calculateHidden2Hidden1() {
//        double[] sigmoidDerivatives = new double[this.numHiddenNeuron2];
        
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            this.deltaHidden1[i] = 0.0;
            for (int j = 0; j < this.numHiddenNeuron2; j++) {
                double sigmoidDerivatives = 
                    Activation.sigmoidDerivative(
                            this.hiddenNeurons2[j].getValue());
                
                this.deltaHidden1[i] += this.deltaHidden2[j] * 
                        this.hiddenNeurons1[i].getMappedValue() * 
                        sigmoidDerivatives;
                
                this.hidden1Hidden2Connections[j][i] += (this.learningRate * 
                        this.deltaHidden2[j] * 
                        sigmoidDerivatives);
            }
        }
        
        
//        for (int i = 0; i < this.numHiddenNeuron2; i++) {
//            
//            sigmoidDerivatives[i] = 
//                    Activation.sigmoidDerivative(
//                            this.hiddenNeurons2[i].getValue());
//            
//            for (int j = 0; j < this.numHiddenNeuron1; j++) {
//                this.hidden1Hidden2Connections[i][j] += 
//                        this.learningRate * (this.crossEntropyDerivatives[i] *
//                        sigmoidDerivatives[i] * 
//                        this.hiddenNeurons1[j].getMappedValue());
//            }
//            
//        }
    }
    
    private void calculateHidden1Input() {
        for (int i = 0; i < this.numInputNeuron; i++) {
            this.deltaInput[i] = 0.0;
            for (int j = 0; j < this.numHiddenNeuron1; j++) {
                double sigmoidDerivatives = 
                    Activation.sigmoidDerivative(
                            this.hiddenNeurons1[j].getValue());
                
                this.deltaInput[i] += this.deltaHidden1[j] * 
                        this.inputNeurons[i].getMappedValue() * 
                        sigmoidDerivatives;
                
                this.inputHidden1Connections[j][i] += (this.learningRate * 
                        this.deltaHidden1[j] * 
                        sigmoidDerivatives);
            }
        }
        
//        double[] sigmoidDerivatives = new double[this.numHiddenNeuron1];
//        for (int i = 0; i < this.numHiddenNeuron1; i++) {
//            
//            sigmoidDerivatives[i] = 
//                    Activation.sigmoidDerivative(
//                            this.hiddenNeurons1[i].getValue());
//            
//            for (int j = 0; j < this.numInputNeuron; j++) {
//                this.inputHidden1Connections[i][j] += 
//                        this.learningRate * (this.crossEntropyDerivatives[i] *
//                        sigmoidDerivatives[i] * 
//                        this.inputNeurons[j].getMappedValue());
//            }
//            
//        }
    }
    
    private double calculateLoss(double[] actual) {
        double[] predicted = new double[this.numOutputNeuron];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            predicted[i] = this.outputNeurons[i].getMappedValue();
        }
        return Loss.crossEntropy(actual, predicted);
    }
    
    private void calculateInputHidden1(double[] data) {
        for (int i = 0; i < this.numInputNeuron; i++) {
            this.inputNeurons[i].setValue(data[i]);
        }
        
        double bias = MathFx.randUniform(1);
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            double totalInputValue = 0.0;
            for (int j = 0; j < this.numInputNeuron; j++) {
                totalInputValue += (this.inputHidden1Connections[i][j] * 
                        this.inputNeurons[j].getValue());
            }
            double mappedValue = Activation.sigmoid(totalInputValue + bias);
//            System.out.println(mappedValue);
            this.hiddenNeurons1[i].setValue(totalInputValue + bias);
            this.hiddenNeurons1[i].setMappedValue(mappedValue);
        }
    }
    
    private void calculateHidden1Hidden2() {
        double bias = MathFx.randUniform(1);
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            double totalInputValue = 0.0;
            for (int j = 0; j < this.numHiddenNeuron1; j++) {
                totalInputValue += (this.hidden1Hidden2Connections[i][j] * 
                        this.hiddenNeurons1[j].getMappedValue());
            }
            double mappedValue = Activation.sigmoid(totalInputValue + bias);
            this.hiddenNeurons2[i].setValue(totalInputValue + bias);
            this.hiddenNeurons2[i].setMappedValue(mappedValue);
        }
    }
    
    private double[] calculateHidden2Output() {
        double bias = MathFx.randUniform(1);
        double[] inputValues = new double[this.numOutputNeuron];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            double totalInputValue = 0.0;
            for (int j = 0; j < this.numHiddenNeuron2; j++) {
                totalInputValue += (this.hidden2OutputConnections[i][j] * 
                        this.hiddenNeurons2[j].getMappedValue());
            }
            inputValues[i] = totalInputValue + bias;
            this.outputNeurons[i].setValue(inputValues[i]);
        }
        
        double[] mappedValues = Activation.softmax(inputValues);
        for (int i = 0; i < mappedValues.length; i++) {
            this.outputNeurons[i].setMappedValue(mappedValues[i]);
        }
        
        return mappedValues;
    }
    
    
    
    private void initializeInputHidden1Connections() {
        this.inputHidden1Connections = 
                new double[this.numHiddenNeuron1][this.numInputNeuron];
        this.deltaInput = new double[this.numInputNeuron];
        this.deltaHidden1 = new double[this.numHiddenNeuron1];
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            for (int j = 0; j < this.numInputNeuron; j++) {
                this.inputHidden1Connections[i][j] = MathFx.randUniform(1);
            }
        }
    }
    
    private void initializeHidden1Hidden2Connections() {
        this.hidden1Hidden2Connections = 
                new double[this.numHiddenNeuron2][this.numHiddenNeuron1];
        this.deltaHidden2 = new double[this.numHiddenNeuron2];
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            for (int j = 0; j < this.numHiddenNeuron1; j++) {
                this.hidden1Hidden2Connections[i][j] = MathFx.randUniform(1);
            }
        }
    }
    
    private void initializeHidden2OutputConnections() {
        this.hidden2OutputConnections =
                new double[this.numOutputNeuron][this.numHiddenNeuron2];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            for (int j = 0; j < this.numHiddenNeuron2; j++) {
                this.hidden2OutputConnections[i][j] = MathFx.randUniform(1);
            }
        }
    }
    
    private void initializeInputNeurons() {
        this.inputNeurons = new Neuron[this.numInputNeuron];
        for (int i = 0; i < this.numInputNeuron; i++) {
            this.inputNeurons[i] = new Neuron(Type.INPUT);
        }
    }
    
    private void initializeHiddenNeurons1() {
        this.hiddenNeurons1 = new Neuron[this.numHiddenNeuron1];
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            this.hiddenNeurons1[i] = new Neuron(Type.HIDDEN);
        }
    }
    
    private void initializeHiddenNeurons2() {
        this.hiddenNeurons2 = new Neuron[this.numHiddenNeuron2];
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            this.hiddenNeurons2[i] = new Neuron(Type.HIDDEN);
        }
    }
    
    private void initializeOutputNeurons() {
        this.outputNeurons = new Neuron[this.numOutputNeuron];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            this.outputNeurons[i] = new Neuron(Type.OUTPUT);
        }
    }
}
