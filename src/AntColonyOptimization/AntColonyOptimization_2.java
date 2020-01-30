/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package AntColonyOptimization;

import Control.MathFx;
import NeuralNetwork.ConfusionMatrix;
import NeuralNetwork.NeuralNetwork;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.List;
import javax.swing.ImageIcon;
import javax.swing.JProgressBar;
import javax.swing.table.DefaultTableModel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;

/**
 *
 * @author Asus
 */
public class AntColonyOptimization_2 {
    
    private int NC;
    private double Q;
    private double P;
    private double target;
    private int m;
    
    private List<Ant_2> ants = new ArrayList<>();
    private List<Double> fitnesses = new ArrayList<>();
    private List<double[][][]> weightMatrices = new ArrayList<>();
    private List<Double> epochLoss;
    
    public AntColonyOptimization_2(int numPopulation, int numIteration, 
            double evaporationRate, double Q, double target) {
        this.m = numPopulation;
        this.NC = numIteration;
        this.P = evaporationRate;
        this.Q = Q;
        this.target = target;
    }
    
    public void initializePopulations(double[][] features, double[][] classes, 
            int numHiddenNeuron, double learningRate, int epoch, 
            double splitRatio) {
        
        for (int i = 0; i < this.m; i++) {
            NeuralNetwork nn = new NeuralNetwork(features, classes, 
                numHiddenNeuron, learningRate, epoch, splitRatio);
            NeuralNetwork backupNn = new NeuralNetwork(features, classes, 
                numHiddenNeuron, learningRate, epoch, splitRatio);
            Ant_2 ant = new Ant_2(nn, backupNn, this.Q, this.P);
            ant.initialize();
            this.ants.add(ant);
        }
        
    }
    
    public NeuralNetwork execute(JProgressBar progressBar, 
            javax.swing.JTable resultTable, 
            javax.swing.JLabel neuralNetworkLossChart) {
        this.epochLoss = new ArrayList<>();
        int progress = 0;
        int currentProgress = 0;
        int maxProgress = this.NC * this.ants.size();
        
        DefaultTableModel model = (DefaultTableModel)resultTable.getModel();
        model.setRowCount(4);
        model.setColumnCount(2);
        
        for (int i = 0; i < this.NC; i++) {
            
            double totalPheromones = 0;
            int fittestIndex = -1;
            double currentMaxFitness = 0;
            this.weightMatrices = new ArrayList<>();
            ConfusionMatrix cmMax = null;
            
            for (int j = 0; j < this.m; j++) {
                ConfusionMatrix cm = this.ants.get(j).launch();
                if (this.ants.get(j).getFitness() > currentMaxFitness) {
                    cmMax = cm;
                    currentMaxFitness = this.ants.get(j).getFitness();
                    fittestIndex = j;
                }
                totalPheromones += this.ants.get(j).calculateTotalPheromones();
                
                currentProgress++;
                progress = (int)(((double)currentProgress / 
                        (double)maxProgress) * 100);
                progressBar.setValue(progress);
                progressBar.setString(progress + "%");
            }
            
            
            System.out.println("CURRENT MAX FITNESS: " + currentMaxFitness);
            if (currentMaxFitness > target) {
                this.ants.get(fittestIndex).getNn().saveWeightOpt();
                model.setValueAt("(Test = " + Math.round(((cmMax.getAccuracy() * 100.0) / 100.0) 
                    * 100.0) + "%)", 0, 1);
                model.setValueAt("(Test = " + Math.round(((cmMax.getPrecision() * 100.0) / 100.0) 
                        * 100.0) + "%)", 1, 1);
                model.setValueAt("(Test = " + Math.round(((cmMax.getRecall() * 100.0) / 100.0) 
                        * 100.0) + "%)", 2, 1);
                model.setValueAt("(Test = " + Math.round(((cmMax.getF1score() * 100.0) / 100.0) 
                        * 100.0) + "%)", 3, 1);

                this.epochLoss.add(this.ants.get(fittestIndex).getNn().getError());
                this.displayLossChart(neuralNetworkLossChart);
                return this.ants.get(fittestIndex).getNn();
            }
            
//            for (int j = 0; j < this.m; j++) {
//                this.ants.get(j).updatePheromones(j == fittestIndex);
//                this.ants.get(j).calculateTransferProbability(totalPheromones);
//                if (this.ants.get(j).getTransferProbability() > MathFx.randUniform(1)) {
//                    this.weightMatrices.add(this.ants.get(j).getPheromoneMatrix());
//                }
//            }
            
//            if (this.weightMatrices.size() > 0) {
//                for (int j = 0; j < this.m; j++) {
//                    double[][][] pheromones = 
//                            this.weightMatrices.get(
//                                    MathFx.randInt(this.weightMatrices.size() - 1));
//                    this.ants.get(j).setBackupWeights(pheromones[0], 
//                            pheromones[1], pheromones[2]);
//                }
//            }

            model.setValueAt("(Test = " + Math.round(((cmMax.getAccuracy() * 100.0) / 100.0) 
                        * 100.0) + "%)", 0, 1);
            model.setValueAt("(Test = " + Math.round(((cmMax.getPrecision() * 100.0) / 100.0) 
                    * 100.0) + "%)", 1, 1);
            model.setValueAt("(Test = " + Math.round(((cmMax.getRecall() * 100.0) / 100.0) 
                    * 100.0) + "%)", 2, 1);
            model.setValueAt("(Test = " + Math.round(((cmMax.getF1score() * 100.0) / 100.0) 
                    * 100.0) + "%)", 3, 1);
            this.epochLoss.add(this.ants.get(fittestIndex).getNn().getError());
            this.displayLossChart(neuralNetworkLossChart);
            
            if (i == this.NC - 1) {
                this.ants.get(fittestIndex).getNn().saveWeightOpt();
                
                return this.ants.get(fittestIndex).getNn();
            }
        }
        
        return null;
    }
    
    public void displayLossChart(javax.swing.JLabel lossChart) {
        DefaultCategoryDataset dataset = new DefaultCategoryDataset();
        for (int i = 0; i < this.epochLoss.size(); i++) {
            dataset.addValue(this.epochLoss.get(i), "Loss", 
                    String.valueOf((i + 1)));
        }
        JFreeChart chart = ChartFactory.createLineChart(
                "Error / Loss Overtime",
                "Time", "Loss",
                dataset,
                PlotOrientation.VERTICAL,
                true, true, false
        );
        
        BufferedImage imageChart = chart.createBufferedImage(
                lossChart.getWidth(), lossChart.getHeight());
        Image im = imageChart;
        lossChart.setIcon(new ImageIcon(im));
        
    }
}
