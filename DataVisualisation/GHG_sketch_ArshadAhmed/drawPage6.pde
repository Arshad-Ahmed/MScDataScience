void drawPage6()
{ 
  //load data
  Table data = loadTable("CAIT_UK_SocioEconominc_GHG_CO2.csv","header,csv");
  
  //extract data columns into arrays
  float[] total_co2 = data.getFloatColumn("Total CO2 Emissions Excluding Land-Use Change and Forestry (MtCO2)");
  float[] pop = data.getFloatColumn("Population (People)"); 
  float[] gdp_usd = data.getFloatColumn("GDP-USD (Million US$ (2005))");
  float[] energy_use = data.getFloatColumn("Energy Use (Thous. tonnes oil eq. (ktoe))");
  float[] year = data.getFloatColumn("Year");
  float[] ghg = data.getFloatColumn("Total GHG Emissions Including Land-Use Change and Forestry (MtCO2)");
  
  String title = "Plot of Yearly Energy Use sized by GDP"; 
  
  //set up scatter plot
  textSize(14);
  ch6 = new XYChart(this);
  ch6.setData(year,energy_use);
  ch6.showXAxis(true);
  ch6.showYAxis(true);
  ch6.setPointColour(color(#3B3131,150));
  ch6.setPointSize(gdp_usd,25);
  ch6.setXAxisLabel("Year");
  ch6.setYAxisLabel("Energy Use (Thous. tonnes oil eq. (ktoe))");
  ch6.setYFormat("###");
  ch6.setXFormat("###");
  ch6.setAxisLabelColour(0);
  ch6.setAxisValuesColour(0);
 
  ch6.draw(20,80,width*0.9,height*0.85);
  
  //title
  fill(#4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);

  text(title, width*0.5,height*0.05);
  rect(285,50,textWidth(title),5);
}