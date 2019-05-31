/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Control;

import Entity.Image;
import NeuralNetwork.NeuralNetwork;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author Azhary Arliansyah
 */
public class NNANT {

    /**
     * @param args the command line arguments
     */
    public static void main(String[] args) {
        // TODO code application logic here
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        final String BASE_DIR = System.getProperty("user.dir");
        
        double height = 10;
        double width = 10;
        double sigma = 1.4;
        double thetaDeg = 135;
        double lambda = 4;
        double gamma = 1.0;
        double psiDeg = 0;
        int ktype;
        
        Size ksize = new Size(height, width);
        
        Mat image = Imgcodecs.imread(BASE_DIR + "/data/erlina/rsz_erlina1.bmp");
//        for (int i = 0; i < image.rows(); i++) {
//            for (int j = 0; j < image.cols(); j++) {
//                System.out.println(Arrays.toString(image.get(i, j)));
//            }
//        }
        
        double theta = thetaDeg * Math.PI / 180;
        double psi = psiDeg * Math.PI / 180;
        
        FileHandler.read("data");
        System.out.println(FileHandler.LABELS);
        Object[] labels = FileHandler.LABELS.keySet().toArray();
        Map<String, double[]> encodedLabels = new HashMap<>();
        for (int i = 0; i < labels.length; i++) {
            double[] encoded = new double[labels.length];
            encoded[i] = 1.0;
            encodedLabels.put((String)labels[i], encoded);
            System.out.println(Arrays.toString(encoded));
        }
        
        List<double[]> features = new ArrayList<>();
        List<double[]> classes = new ArrayList<>();
        List<Image> images = new ArrayList<>();
        Mat kernel = Imgproc.getGaborKernel(ksize, sigma, theta, lambda, gamma);
        for (Map.Entry<String, List<String>> ent: 
                FileHandler.LABELS.entrySet()) {
            String path = "data/" + ent.getKey();
            for (String filename : ent.getValue()) {
                Image img = new Image(path + "/" + filename, kernel, 
                        ent.getKey());
                images.add(img);
                
            }
        }
        
        Collections.shuffle(images);
        for (Image img : images) {
            features.add(img.getFilteredData());
            classes.add(encodedLabels.get(img.getLabel()));
        }
        
        double[][] finalFeatures = new double[features.size()][];
        double[][] finalClasses = new double[classes.size()][];
        
        for (int i = 0; i < features.size(); i++) {
            finalFeatures[i] = features.get(i);
            finalClasses[i] = classes.get(i);
        }
        
        NeuralNetwork nn = new NeuralNetwork(finalFeatures, finalClasses, 
                labels.length + 2);
        nn.fit();
        
//        Mat mat1 = new Mat(image.rows(), image.cols(), CvType.CV_8UC1);
//        
//        
//        Imgproc.cvtColor(image, mat1, Imgproc.COLOR_RGB2GRAY);
//        
        
//        Mat dest = new Mat(mat1.rows(), mat1.cols(), image.type());
//        Imgproc.filter2D(mat1, dest, mat1.type(), kernel);
//        
//        // Save the result.
//        String filename = "gaborFiltered.png";
//        System.out.println(String.format("Writing %s", filename));
//        Imgcodecs.imwrite(filename, dest);
    }
    
}
