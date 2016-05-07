// Draws page 2 of the sketch.
 
void drawPage2()
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
  
  String title = "Plot of Total C02 Emissions vs Energy Use sized by Population"; 
  
  //set up scatter plot
  textSize(14);
  ch2 = new XYChart(this);
  ch2.setData(energy_use, total_co2);
  ch2.showXAxis(true);
  ch2.showYAxis(true);
  ch2.setPointColour(color(#7D0552,150));
  ch2.setPointSize(pop,25);
  ch2.setXAxisLabel("Energy Use (Thous. tonnes oil eq. (ktoe))");
  ch2.setYAxisLabel("Total CO2 Emissions Excluding Land-Use Change and Forestry (MtCO2)");
  ch2.setYFormat("###");
  ch2.setXFormat("###");
  ch2.setAxisLabelColour(0);
  ch2.setAxisValuesColour(0);
  
  ch2.draw(20,80,width*0.9,height*0.85);
  
  //title
  fill(#4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5,height*0.05);
  
  rect(166,50,textWidth(title),5);

 
}