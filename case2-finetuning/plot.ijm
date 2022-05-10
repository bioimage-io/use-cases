macro "Scatter Plot" {
        Plot.create("Scatter Plot", "X", "Y");
        x = Table.getColumn("Volume");
        y = Table.getColumn("Sphericity");
        Plot.add("dots", x, y);
        Plot.setAxisLabelSize(18.0, "bold");
		Plot.setFontSize(18.0);
		Plot.setXYLabels("Volume", "Sphericity");
		Plot.setFormatFlags("11001100001111");
		Plot.setLimits(-15036,600000,-0.05,2.00);
        Plot.setStyle(0, "orange,none,2.0,Dot");
        
  }
 