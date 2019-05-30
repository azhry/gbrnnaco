/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NeuralNetwork;

import NeuralNetwork.Neuron.Type;
import Control.MathFx;

/**
 *
 * @author Azhary Arliansyah
 */
public class NeuralNetwork {
    
    private final double learningRate = 0.01;
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
    
    private double[] crossEntropyDerivatives;
    private double[][] data;
    private double[][] target;
    
    private double loss;
    
    public NeuralNetwork(double[][] data, double[][] target, 
            int numHiddens, int numOutputs) {
        this.data = data;
        this.target = target;
        
        this.numInputNeuron = this.data[0].length;
        this.numHiddenNeuron1 = numHiddens;
        this.numHiddenNeuron2 = numHiddens;
        this.numOutputNeuron = numOutputs;
        
        this.initializeInputNeurons();
        this.initializeHiddenNeurons1();
        this.initializeHiddenNeurons2();
        this.initializeOutputNeurons();
    
        this.initializeInputHidden1Connections();
        this.initializeHidden1Hidden2Connections();
        this.initializeHidden2OutputConnections();
    }
    
    public void fit() {
        for (int i = 0; i < this.data.length; i++) {
            this.feedforward(this.data[i]);
            System.out.println("Loss: " + this.calculateLoss(this.target[i]));
            this.backpropagation(this.target[i]);
        }
    }
    
    public void predict() {
        
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
        
        for (int i = 0; i < this.numOutputNeuron; i++) {
            for (int j = 0; j < this.numHiddenNeuron2; j++) {
                this.hidden2OutputConnections[i][j] -= 
                        (this.learningRate * (this.crossEntropyDerivatives[i] * 
                        softmaxDerivatives[i] *
                        this.hiddenNeurons2[j].getMappedValue()));
            }
        }
    }
    
    private void calculateHidden2Hidden1() {
        double[] sigmoidDerivatives = new double[this.numHiddenNeuron2];
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            
            sigmoidDerivatives[i] = 
                    Activation.sigmoidDerivative(
                            this.hiddenNeurons2[i].getValue());
            
            for (int j = 0; j < this.numHiddenNeuron1; j++) {
                this.hidden1Hidden2Connections[i][j] = 
                        this.learningRate * (this.crossEntropyDerivatives[i] *
                        sigmoidDerivatives[i] * 
                        this.hiddenNeurons1[j].getMappedValue());
            }
            
        }
    }
    
    private void calculateHidden1Input() {
        double[] reluDerivatives = new double[this.numHiddenNeuron1];
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            
            reluDerivatives[i] = 
                    Activation.reluDerivative(
                            this.hiddenNeurons1[i].getValue());
            
            for (int j = 0; j < this.numInputNeuron; j++) {
                this.inputHidden1Connections[i][j] = 
                        this.learningRate * (this.crossEntropyDerivatives[i] *
                        reluDerivatives[i] * 
                        this.inputNeurons[j].getMappedValue());
            }
            
        }
    }
    
    private double calculateLoss(double[] actual) {
        double[] predicted = new double[this.numOutputNeuron];
        for (int i = 0; i < this.numOutputNeuron; i++) {
            predicted[i] = this.outputNeurons[i].getValue();
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
            double mappedValue = Activation.relu(totalInputValue + bias);
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
    
    private void calculateHidden2Output() {
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
    }
    
    
    
    private void initializeInputHidden1Connections() {
        this.inputHidden1Connections = 
                new double[this.numHiddenNeuron1][this.numInputNeuron];
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            for (int j = 0; j < this.numInputNeuron; j++) {
                this.inputHidden1Connections[i][j] = MathFx.randUniform(1);
            }
        }
    }
    
    private void initializeHidden1Hidden2Connections() {
        this.hidden1Hidden2Connections = 
                new double[this.numHiddenNeuron2][this.numHiddenNeuron1];
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
