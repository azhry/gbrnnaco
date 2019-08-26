/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Entity;

import java.awt.Image;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;
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
    private Mat rawImgData;
    private double[] filteredData;
    private double[] data;
    private String label;
    private double avgFilter;
    
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
        double total = 0;
        for (int j = 0; j < this.filteredImg.rows(); j++) {
            for (int k = 0; k < this.filteredImg.cols(); k++) {
                for (int l = 0; l < this.filteredImg.channels(); l++) {
                    this.filteredData[i] = this.filteredImg.get(j, k)[l];
                    total += this.filteredImg.get(j, k)[l];
                    i++;
                } 
            }
        }
        
        this.avgFilter = total / (double)(this.filteredImg.rows() * 
                this.filteredImg.cols() * this.filteredImg.channels());
    }
    
    public ImageData(String path, String label) {
        this.label = label;
        this.img = Imgcodecs.imread(path);
        this.rawImg = new BufferedImage(this.img.width(), this.img.height(), 
                BufferedImage.TYPE_INT_ARGB);
        
        this.grayImg = new Mat(this.img.rows(), this.img.cols(), 
                CvType.CV_8UC1);
        Imgproc.cvtColor(this.img, this.grayImg, Imgproc.COLOR_RGB2GRAY);
        
        this.rawImgData = new Mat(this.grayImg.rows(), this.grayImg.cols(), 
                this.img.type());
        
        this.data = new double[this.rawImgData.rows() * 
                this.rawImgData.cols() * this.rawImgData.channels()];
        int i = 0;
        for (int j = 0; j < this.rawImgData.rows(); j++) {
            for (int k = 0; k < this.rawImgData.cols(); k++) {
                for (int l = 0; l < this.rawImgData.channels(); l++) {
                    this.data[i] = this.rawImgData.get(j, k)[l];
                    i++;
                } 
            }
        }
    }
    
    public void filterImg(Mat kernel) {
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
    
    public BufferedImage getRawImg() {
        return this.rawImg;
    }
    
    public Mat getGrayImg() {
        return this.grayImg;
    }
    
    public double getAvgFilter() {
        return this.avgFilter;
    }
    
    public Image getBufferFilteredImg() {
        byte[] data = new byte[this.filteredImg.width() * 
                this.filteredImg.height() * (int)this.filteredImg.elemSize()];
        this.filteredImg.get(0, 0, data);
        int type;
        if (this.filteredImg.channels() == 1) {
            type = BufferedImage.TYPE_BYTE_GRAY;
        }
        else {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        
        BufferedImage im = new BufferedImage(this.filteredImg.width(), 
                this.filteredImg.height(), type);
        im.getRaster().setDataElements(0, 0, this.filteredImg.width(), 
                this.filteredImg.height(), data);
        
        int i = 0;
        double total = 0;
        for (int j = 0; j < this.filteredImg.rows(); j++) {
            for (int k = 0; k < this.filteredImg.cols(); k++) {
                for (int l = 0; l < this.filteredImg.channels(); l++) {
                    this.filteredData[i] = this.filteredImg.get(j, k)[l];
                    total += this.filteredImg.get(j, k)[l];
                    i++;
                } 
            }
        }
        
        this.avgFilter = total / (double)(this.filteredImg.rows() * 
                this.filteredImg.cols() * this.filteredImg.channels());
        
        return im;
    }
    
    public Mat getFilteredImg() {
        return this.filteredImg;
    }
    
    public double[] getFilteredData() {
        return this.filteredData;
    }
    
    public double[] getData() {
        return this.data;
    }
    
    public String getLabel() {
        return this.label;
    }
}
