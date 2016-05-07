void drawPage7()
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
  
  String title = "Plot of Yearly Total CO2 Emissions sized by GDP"; 
  
  //set up scatter plot
  textSize(14);
  ch7 = new XYChart(this);
  ch7.setData(year,total_co2);
  ch7.showXAxis(true);
  ch7.showYAxis(true);
  ch7.setPointColour(color(#41A317,150));
  ch7.setPointSize(gdp_usd,25);
  ch7.setXAxisLabel("Year");
  ch7.setYAxisLabel("Total CO2 Emissions Excluding Land-Use Change and Forestry (MtCO2)");
  ch7.setYFormat("###");
  ch7.setXFormat("###");
  ch7.setAxisLabelColour(0);
  ch7.setAxisValuesColour(0);
 
  ch7.draw(20,80,width*0.9,height*0.85);
  
  //title
  fill(#4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5,height*0.05);
}