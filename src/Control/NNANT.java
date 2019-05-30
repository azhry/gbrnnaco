/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Control;

import org.opencv.core.Core;
import org.opencv.core.CvType;
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
        double theta = thetaDeg * Math.PI / 180;
        double psi = psiDeg * Math.PI / 180;
        
        Mat mat1 = new Mat(image.rows(), image.cols(), CvType.CV_8UC1);
        Imgproc.cvtColor(image, mat1, Imgproc.COLOR_RGB2GRAY);
        
        Mat kernel = Imgproc.getGaborKernel(ksize, sigma, theta, lambda, gamma);
        Mat dest = new Mat(mat1.rows(), mat1.cols(), image.type());
        Imgproc.filter2D(mat1, dest, mat1.type(), kernel);
        
        // Save the result.
        String filename = "gaborFiltered.png";
        System.out.println(String.format("Writing %s", filename));
        Imgcodecs.imwrite(filename, dest);
    }
    
}
