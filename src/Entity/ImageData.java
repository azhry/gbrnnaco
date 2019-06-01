/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Entity;

import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author Azhary Arliansyah
 */
public class ImageData {
    
    private final BufferedImage rawImg;
    private final Mat img;
    
    private Mat grayImg;
    private Mat filteredImg;
    private double[] filteredData;
    private String label;
    
    public ImageData(String path, Mat kernel, String label) {
        this.label = label;
        this.img = Imgcodecs.imread(path);
        this.rawImg = new BufferedImage(this.img.width(), this.img.height(), 
                BufferedImage.TYPE_INT_ARGB);
        this.grayImg = new Mat(this.img.rows(), this.img.cols(), 
                CvType.CV_8UC1);
        Imgproc.cvtColor(this.img, this.grayImg, Imgproc.COLOR_RGB2GRAY);
        
        this.filteredImg = new Mat(this.grayImg.rows(), this.grayImg.cols(), 
                this.img.type());
        Imgproc.filter2D(this.grayImg, this.filteredImg, this.grayImg.type(), 
                kernel);
        
        this.filteredData = new double[this.filteredImg.rows() * 
                this.filteredImg.cols()];
        int i = 0;
        for (int j = 0; j < this.filteredImg.rows(); j++) {
            for (int k = 0; k < this.filteredImg.cols(); k++) {
                for (int l = 0; l < this.filteredImg.channels(); l++) {
                    this.filteredData[i] = this.filteredImg.get(j, k)[l];
                    i++;
                } 
            }
        }
    }
    
    public ImageData(String path, String label) {
        this.label = label;
        System.out.println(path);
        this.img = Imgcodecs.imread(path);
        this.rawImg = new BufferedImage(this.img.width(), this.img.height(), 
                BufferedImage.TYPE_INT_ARGB);
    }
    
    public Mat getImg() {
        return this.img;
    }
    
    public BufferedImage getRawImg() {
        return this.rawImg;
    }
    
    public Mat getGrayImg() {
        return this.grayImg;
    }
    
    public Mat getFilteredImg() {
        return this.filteredImg;
    }
    
    public double[] getFilteredData() {
        return this.filteredData;
    }
    
    public String getLabel() {
        return this.label;
    }
}