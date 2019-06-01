/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package Boundary;

/**
 *
 * @author Azhary Arliansyah
 */
public class ImageRow extends javax.swing.JPanel {

    /**
     * Creates new form ImageRow
     */
    public ImageRow() {
        initComponents();
    }

    /**
     * This method is called from within the constructor to initialize the form.
     * WARNING: Do NOT modify this code. The content of this method is always
     * regenerated by the Form Editor.
     */
    @SuppressWarnings("unchecked")
    // <editor-fold defaultstate="collapsed" desc="Generated Code">//GEN-BEGIN:initComponents
    private void initComponents() {

        jPanel5 = new javax.swing.JPanel();
        jLabel11 = new javax.swing.JLabel();
        filename = new javax.swing.JLabel();
        rawImage = new javax.swing.JLabel();
        filteredImage = new javax.swing.JLabel();
        jLabel13 = new javax.swing.JLabel();
        filelabel = new javax.swing.JLabel();

        jPanel5.setBorder(javax.swing.BorderFactory.createLineBorder(new java.awt.Color(0, 0, 0)));
        jPanel5.setPreferredSize(new java.awt.Dimension(558, 100));
        jPanel5.setLayout(new org.netbeans.lib.awtextra.AbsoluteLayout());

        jLabel11.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        jLabel11.setText("Filename");
        jPanel5.add(jLabel11, new org.netbeans.lib.awtextra.AbsoluteConstraints(20, 20, -1, 9));

        filename.setText("h3h3.png");
        jPanel5.add(filename, new org.netbeans.lib.awtextra.AbsoluteConstraints(30, 30, -1, -1));

        rawImage.setFont(new java.awt.Font("Tahoma", 0, 24)); // NOI18N
        rawImage.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        rawImage.setIcon(new javax.swing.ImageIcon(getClass().getResource("/Icon/setting.png"))); // NOI18N
        jPanel5.add(rawImage, new org.netbeans.lib.awtextra.AbsoluteConstraints(190, 0, 160, 120));

        filteredImage.setFont(new java.awt.Font("Tahoma", 0, 24)); // NOI18N
        filteredImage.setHorizontalAlignment(javax.swing.SwingConstants.CENTER);
        filteredImage.setIcon(new javax.swing.ImageIcon(getClass().getResource("/Icon/setting.png"))); // NOI18N
        jPanel5.add(filteredImage, new org.netbeans.lib.awtextra.AbsoluteConstraints(390, 0, 160, 120));

        jLabel13.setFont(new java.awt.Font("Tahoma", 1, 11)); // NOI18N
        jLabel13.setText("Label");
        jPanel5.add(jLabel13, new org.netbeans.lib.awtextra.AbsoluteConstraints(20, 70, -1, 9));

        filelabel.setText("azhry");
        jPanel5.add(filelabel, new org.netbeans.lib.awtextra.AbsoluteConstraints(30, 80, -1, -1));

        javax.swing.GroupLayout layout = new javax.swing.GroupLayout(this);
        this.setLayout(layout);
        layout.setHorizontalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel5, javax.swing.GroupLayout.PREFERRED_SIZE, javax.swing.GroupLayout.DEFAULT_SIZE, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );
        layout.setVerticalGroup(
            layout.createParallelGroup(javax.swing.GroupLayout.Alignment.LEADING)
            .addGroup(layout.createSequentialGroup()
                .addComponent(jPanel5, javax.swing.GroupLayout.PREFERRED_SIZE, 122, javax.swing.GroupLayout.PREFERRED_SIZE)
                .addGap(0, 0, Short.MAX_VALUE))
        );
    }// </editor-fold>//GEN-END:initComponents


    // Variables declaration - do not modify//GEN-BEGIN:variables
    public javax.swing.JLabel filelabel;
    public javax.swing.JLabel filename;
    public javax.swing.JLabel filteredImage;
    private javax.swing.JLabel jLabel11;
    private javax.swing.JLabel jLabel13;
    private javax.swing.JPanel jPanel5;
    public javax.swing.JLabel rawImage;
    // End of variables declaration//GEN-END:variables
}
