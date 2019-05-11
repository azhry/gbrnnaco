/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NeuralNetwork;

import NeuralNetwork.Neuron.Type;
import nn.ant.MathFx;

/**
 *
 * @author Azhary Arliansyah
 */
public class NeuralNetwork {
    
    private int numInputNeuron;
    private int numHiddenNeuron1;
    private int numHiddenNeuron2;
    private int numOutputNeuron;
    
    private Neuron[] inputNeurons;
    private Neuron[] hiddenNeurons1;
    private Neuron[] hiddenNeurons2;
    private Neuron[] outputNeurons;
    
    private double[][] inputHidden1Connections;
    private double[][] hidden1Hidden2Connections;
    private double[][] hidden2OutputConnections;
    
    public NeuralNetwork(int numInputs, int numHiddens, int numOutputs) {
        this.numInputNeuron = numInputs;
        this.numHiddenNeuron1 = numHiddens;
        this.numHiddenNeuron2 = numHiddens;
        this.numOutputNeuron = numOutputs;
        
        this.initializeInputNeurons();
        this.initializeHiddenNeurons1();
        this.initializeOutputNeurons();
    
        this.initializeInputHidden1Connections();
        this.initializeHidden2OutputConnections();
    }
    
    public void fit() {
        
    }
    
    public void predict() {
        
    }
    
    private void calculateInputHidden1() {
        double bias = MathFx.randUniform(1);
        for (int i = 0; i < this.numHiddenNeuron1; i++) {
            double totalInputValue = 0.0;
            for (int j = 0; j < this.numInputNeuron; j++) {
                totalInputValue += (this.inputHidden1Connections[i][j] * 
                        this.inputNeurons[j].getValue());
            }
            double mappedValue = Activation.relu(totalInputValue + bias);
            this.hiddenNeurons1[i].setValue(mappedValue);
        }
    }
    
    private void calculateHidden1Hidden2() {
        double bias = MathFx.randUniform(1);
        for (int i = 0; i < this.numHiddenNeuron2; i++) {
            double totalInputValue = 0.0;
            for (int j = 0; j < this.numHiddenNeuron1; j++) {
                totalInputValue += (this.hidden1Hidden2Connections[i][j] * 
                        this.hiddenNeurons1[j].getValue());
            }
            double mappedValue = Activation.sigmoid(totalInputValue + bias);
            this.hiddenNeurons2[i].setValue(mappedValue);
        }
    }
    
    private void calculateHidden2Output() {
        double bias = MathFx.randUniform(1);
        for (int i = 0; i < this.numOutputNeuron; i++) {
            double totalHiddenValue = 0.0;
            for (int j = 0; j < this.numHiddenNeuron2; j++) {
                totalHiddenValue += (this.hidden2OutputConnections[i][j] * 
                        this.hiddenNeurons2[j].getValue());
            }
            this.outputNeurons[i].calculateValue(totalHiddenValue + bias);
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
