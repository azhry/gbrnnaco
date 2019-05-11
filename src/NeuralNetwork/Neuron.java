/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package NeuralNetwork;

import nn.ant.MathFx;

/**
 *
 * @author Azhary Arliansyah
 */
public class Neuron {
    
    public enum Type {
        INPUT, HIDDEN, OUTPUT, BIAS
    };
    
    private double weight;
    private double value;
    private Type type;
    
    public Neuron(double weight, Type type) {
        this.weight = weight;
        this.type = type;
    }
    
    public Neuron(Type type) {
        this.type = type;
        this.weight = MathFx.randUniform(1);
    }
    
    public void setWeight(double weight) {
        this.weight = weight;
    }
    
    public void setValue(double value) {
        this.value = value;
    }
    
    public void setType(Type type) {
        this.type = type;
    }
    
    public void calculateValue(double inputValue) {
        this.value = this.sigmoid(inputValue);
    }
    
    public double getValue() {
        return this.value;
    }
    
    private double sigmoid(double value) {
        return 1 / (1 + Math.pow(Math.E, -value));
    }
    
    private double softmax(double value) {
        return value;
    }
}
