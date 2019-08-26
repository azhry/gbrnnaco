/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package AntColonyOptimization;

import Control.MathFx;
import Control.Reflector;
import NeuralNetwork.ConfusionMatrix;
import NeuralNetwork.NeuralNetwork;
import java.awt.Image;
import java.awt.image.BufferedImage;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import javax.swing.ImageIcon;
import javax.swing.JProgressBar;
import javax.swing.table.DefaultTableModel;
import org.jfree.chart.ChartFactory;
import org.jfree.chart.JFreeChart;
import org.jfree.chart.plot.PlotOrientation;
import org.jfree.data.category.DefaultCategoryDataset;


/**
 *
 * @author Azhary Arliansyah
 */
public class AntColonyOptimization {
    
    private final String[] layerConnections = new String[] {
        "InputHidden1", "Hidden1Hidden2", "Hidden2Output"
    };
    
    private final int numberOfAnts;
    private final double Q; // quantity of deposit pheromone
    private final double alpha; // pheromone factor
    private final double beta; // heuristics factor
    private final double q0; // a uniformly distributed number
    
    private List<Map<String, double[][]>> vertices;
    private final List<Map<String, Double>> trails; // store pheromones and probabilities
    private final NeuralNetwork nn;

    private final int numSolution;
    private final int numIteration;
    private Ant[] ants;
    
    private int FITTEST_INDEX = -1;
    private double HIGHEST_ACCURACY = 0;
    private ConfusionMatrix FITTEST_CM = new ConfusionMatrix();
    
    private List<Double> epochLoss;
    
    public AntColonyOptimization(NeuralNetwork nn, 
            int numberOfAnts, double Q, double alpha, double beta, int iteration) {
        this.numberOfAnts = numberOfAnts;
        this.Q = Q;
        this.alpha = alpha;
        this.beta = beta;
        this.q0 = MathFx.randUniform(1);
        this.vertices = new ArrayList<>();
        this.trails = new ArrayList<>();
        this.nn = nn;
        this.numSolution = numberOfAnts;
        this.ants = new Ant[this.numberOfAnts];
        this.FITTEST_INDEX = -1;
        this.HIGHEST_ACCURACY = 0;
        this.FITTEST_CM = new ConfusionMatrix();
        this.numIteration = iteration;
        
        this.epochLoss = new ArrayList<>();

        this.generateVertices();
        this.generateAnts();
    }
    
    public void executeAco(double[][] features, double[][] classes, 
            int numHiddenNeuron, double learningRate, int epoch, 
            double splitRatio) {
        
        for (int j = 0; j < this.numIteration; j++) {
            for (int i = 0; i < this.ants.length; i++) {
            
                ConfusionMatrix cm = this.ants[i].move(features, classes, 
                        numHiddenNeuron, learningRate, epoch, splitRatio);

                double accuracy = cm.getAccuracy();
                if (accuracy > this.HIGHEST_ACCURACY) {
                    this.FITTEST_CM = cm;
                    this.HIGHEST_ACCURACY = accuracy;
                    this.FITTEST_INDEX = i;
                }

            }
        }
        
        this.ants[this.FITTEST_INDEX].getNn().saveWeightOpt();
    }
    
    public void executeAco(double[][] features, double[][] classes, 
            int numHiddenNeuron, double learningRate, int epoch, 
            double splitRatio, JProgressBar progressBar, 
            javax.swing.JTable resultTable) {
        
        int progress = 0;
        int currentProgress = 0;
        int maxProgress = this.numIteration * this.ants.length;

        DefaultTableModel model = (DefaultTableModel)resultTable.getModel();
        model.setRowCount(4);
        model.setColumnCount(2);

        for (int j = 0; j < this.numIteration; j++) {
            for (int i = 0; i < this.ants.length; i++) {

                ConfusionMatrix cm = this.ants[i].move(features, classes, 
                        numHiddenNeuron, learningRate, epoch, splitRatio);

                double accuracy = cm.getAccuracy();
                if (accuracy > this.HIGHEST_ACCURACY) {
                    this.FITTEST_CM = cm;
                    this.HIGHEST_ACCURACY = accuracy;
                    this.FITTEST_INDEX = i;
                }

                currentProgress++;
                progress = (int)(((double)currentProgress / 
                        (double)maxProgress) * 100);
                progressBar.setValue(progress);
                progressBar.setString(progress + "%");

                model.setValueAt("(Train = " + Math.round(((cm.getAccuracy() * 100.0) / 100.0) 
                        * 100.0) + "%)", 0, 1);
                model.setValueAt("(Train = " + Math.round(((cm.getPrecision() * 100.0) / 100.0) 
                        * 100.0) + "%)", 1, 1);
                model.setValueAt("(Train = " + Math.round(((cm.getRecall() * 100.0) / 100.0) 
                        * 100.0) + "%)", 2, 1);
                model.setValueAt("(Train = " + Math.round(((cm.getF1score() * 100.0) / 100.0) 
                        * 100.0) + "%)", 3, 1);
            }
        }
        
        
        this.ants[this.FITTEST_INDEX].getNn().saveWeightOpt();
        ConfusionMatrix cm = this.FITTEST_CM;
        model.setValueAt("(Train = " + Math.round(((cm.getAccuracy() * 100.0) / 100.0) 
                    * 100.0) + "%)", 0, 1);
        model.setValueAt("(Train = " + Math.round(((cm.getPrecision() * 100.0) / 100.0) 
                * 100.0) + "%)", 1, 1);
        model.setValueAt("(Train = " + Math.round(((cm.getRecall() * 100.0) / 100.0) 
                * 100.0) + "%)", 2, 1);
        model.setValueAt("(Train = " + Math.round(((cm.getF1score() * 100.0) / 100.0) 
                * 100.0) + "%)", 3, 1);
        this.ants[this.FITTEST_INDEX].getNn().scoreOpt(features, classes, model);
        System.out.println("END ACO");
    }
    
    public void executeAco(double[][] features, double[][] classes, 
            int numHiddenNeuron, double learningRate, int epoch, 
            double splitRatio, JProgressBar progressBar, 
            javax.swing.JTable resultTable, 
            javax.swing.JLabel neuralNetworkLossChart) {
        
        int progress = 0;
        int currentProgress = 0;
        int maxProgress = this.numIteration * this.ants.length;

        DefaultTableModel model = (DefaultTableModel)resultTable.getModel();
        model.setRowCount(4);
        model.setColumnCount(2);

        for (int j = 0; j < this.numIteration; j++) {
            for (int i = 0; i < this.ants.length; i++) {

                ConfusionMatrix cm = this.ants[i].move(features, classes, 
                        numHiddenNeuron, learningRate, epoch, splitRatio);

                double accuracy = cm.getAccuracy();
                if (accuracy > this.HIGHEST_ACCURACY) {
                    this.FITTEST_CM = cm;
                    this.HIGHEST_ACCURACY = accuracy;
                    this.FITTEST_INDEX = i;
                }

                currentProgress++;
                progress = (int)(((double)currentProgress / 
                        (double)maxProgress) * 100);
                progressBar.setValue(progress);
                progressBar.setString(progress + "%");

                model.setValueAt("(Train = " + Math.round(((cm.getAccuracy() * 100.0) / 100.0) 
                        * 100.0) + "%)", 0, 1);
                model.setValueAt("(Train = " + Math.round(((cm.getPrecision() * 100.0) / 100.0) 
                        * 100.0) + "%)", 1, 1);
                model.setValueAt("(Train = " + Math.round(((cm.getRecall() * 100.0) / 100.0) 
                        * 100.0) + "%)", 2, 1);
                model.setValueAt("(Train = " + Math.round(((cm.getF1score() * 100.0) / 100.0) 
                        * 100.0) + "%)", 3, 1);
                
                this.epochLoss.add(this.ants[i].getNn().getError());
                this.displayLossChart(neuralNetworkLossChart);
            }
        }
        
        
        this.ants[this.FITTEST_INDEX].getNn().saveWeightOpt();
        ConfusionMatrix cm = this.FITTEST_CM;
        model.setValueAt("(Train = " + Math.round(((cm.getAccuracy() * 100.0) / 100.0) 
                    * 100.0) + "%)", 0, 1);
        model.setValueAt("(Train = " + Math.round(((cm.getPrecision() * 100.0) / 100.0) 
                * 100.0) + "%)", 1, 1);
        model.setValueAt("(Train = " + Math.round(((cm.getRecall() * 100.0) / 100.0) 
                * 100.0) + "%)", 2, 1);
        model.setValueAt("(Train = " + Math.round(((cm.getF1score() * 100.0) / 100.0) 
                * 100.0) + "%)", 3, 1);
        this.ants[this.FITTEST_INDEX].getNn().scoreOpt(features, classes, model);
        System.out.println("END ACO");
    }

    private void generateAnts() {
    	for (int i = 0; i < this.ants.length; i++) {
    		this.ants[i] = new Ant(this.Q, this.vertices.get(i));
    	}
    }
    
    private void generateVertices() {
        
        this.vertices = new ArrayList<>();
        for (int i = 0; i < this.numSolution; i++) {
        	Map<String, double[][]> vertex = new HashMap<>();
        	for (String connection : this.layerConnections) {
	            vertex.put(connection, 
	                    (double[][])Reflector.callUserFunc(NeuralNetwork.class, 
	                            this.nn, 
	                            "initialize" + connection + "Connections"));
	        }
	        this.vertices.add(vertex);
        }
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
