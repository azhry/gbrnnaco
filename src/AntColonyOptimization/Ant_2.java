/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package AntColonyOptimization;

import NeuralNetwork.ConfusionMatrix;
import NeuralNetwork.NeuralNetwork;

/**
 *
 * @author Asus
 */
public class Ant_2 {
    
    private NeuralNetwork nn;
    private NeuralNetwork backupNn;
    private double[][] pheromones_ih;
    private double[][] pheromones_hh;
    private double[][] pheromones_ho;
    
    private double[][][] pheromoneMatrix;
    
    private double evaporationRate;
    private double changingRate;
    private double Q;
    
    private double fitness;
    private double transferProbability;
    
    public Ant_2(NeuralNetwork nn, NeuralNetwork backupNn, double Q, 
            double evaporationRate) {
        this.nn = nn;
        this.backupNn = backupNn;
        this.Q = Q;
        this.evaporationRate = evaporationRate;
    }
    
    public double[][][] initialize() {
        this.pheromones_ih = this.nn.getInputHidden1Connections();
        this.pheromones_hh = this.nn.getHidden1Hidden2Connections();
        this.pheromones_ho = this.nn.getHidden2OutputConnections();
        
        this.backupNn.setInputHidden1Connections(this.pheromones_ih);
        this.backupNn.setHidden1Hidden2Connections(this.pheromones_hh);
        this.backupNn.setHidden2OutputConnections(this.pheromones_ho);
        
        this.pheromoneMatrix = new double[][][] {
            this.pheromones_ih, this.pheromones_hh, this.pheromones_ho
        };
        
        return this.pheromoneMatrix;
    }
    
    public double[][][] getPheromoneMatrix() {
        return this.pheromoneMatrix;
    }
    
    private void calculateChangingRate(double errorRate) {
        this.changingRate = this.Q / errorRate;
    }
    
    public void setBackupWeights(double[][] ih, double[][] hh, double[][] ho) {
        this.backupNn.setInputHidden1Connections(ih);
        this.backupNn.setHidden1Hidden2Connections(hh);
        this.backupNn.setHidden2OutputConnections(ho);
    }
    
    public ConfusionMatrix launch() {
        this.nn.setInputHidden1Connections(this.backupNn
                .getInputHidden1Connections());
        this.nn.setHidden1Hidden2Connections(this.backupNn
                .getHidden1Hidden2Connections());
        this.nn.setHidden2OutputConnections(this.backupNn
                .getHidden2OutputConnections());
        
        ConfusionMatrix cm = this.nn.fitOpt();
        this.fitness = cm.getAccuracy();
        return cm;
    }
    
    public NeuralNetwork getNn() {
        return this.nn;
    }
    
    public double getFitness() {
        return this.fitness;
    }
    
    public double calculateTotalPheromones() {
        double result = 0;
        
        for (int i = 0; i < this.pheromones_ih.length; i++) {
            for (int j = 0; j < this.pheromones_ih[i].length; j++) {
                result += this.pheromones_ih[i][j];
            }
        }
        
        for (int i = 0; i < this.pheromones_hh.length; i++) {
            for (int j = 0; j < this.pheromones_hh[i].length; j++) {
                result += this.pheromones_hh[i][j];
            }
        }
        
        for (int i = 0; i < this.pheromones_ho.length; i++) {
            for (int j = 0; j < this.pheromones_ho[i].length; j++) {
                result += this.pheromones_ho[i][j];
            }
        }
        
        return result;
    }
    
    public void calculateTransferProbability(double pheromonesPopulationTotal) {
        this.transferProbability = this.calculateTotalPheromones() / 
                pheromonesPopulationTotal;
    }
    
    public double getTransferProbability() {
        return this.transferProbability;
    }
    
    public void updatePheromones(boolean optimum) {
        double error = this.nn.getError();
        this.calculateChangingRate(error);
        
        for (int i = 0; i < this.pheromones_ih.length; i++) {
            for (int j = 0; j < this.pheromones_ih[i].length; j++) {
                this.pheromones_ih[i][j] = 
                        this.updatePheromone(this.pheromones_ih[i][j], error, 
                                optimum);
            }
        }
        
        for (int i = 0; i < this.pheromones_hh.length; i++) {
            for (int j = 0; j < this.pheromones_hh[i].length; j++) {
                this.pheromones_hh[i][j] = 
                        this.updatePheromone(this.pheromones_hh[i][j], error, 
                                optimum);
            }
        }
        
        for (int i = 0; i < this.pheromones_ho.length; i++) {
            for (int j = 0; j < this.pheromones_ho[i].length; j++) {
                this.pheromones_ho[i][j] = 
                        this.updatePheromone(this.pheromones_ho[i][j], error, 
                                optimum);
            }
        }
    }
    
    private double updatePheromone(double pheromone, double errorRate, 
            boolean optimum) {
        double result = (1 - this.evaporationRate) * pheromone;
        if (optimum) {
            result += (this.evaporationRate * this.changingRate);
        }
        return result;
    }
}
