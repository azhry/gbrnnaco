/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Entity;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

/**
 *
 * @author Azhary Arliansyah
 */
public class Image {
    
    private final Mat img;
    private final Mat grayImg;
    private final Mat filteredImg;
    private final double[] filteredData;
    private String label;
    
    public Image(String path, Mat kernel, String label) {
        this.label = label;
        this.img = Imgcodecs.imread(path);
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
    
    public Mat getImg() {
        return this.img;
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
