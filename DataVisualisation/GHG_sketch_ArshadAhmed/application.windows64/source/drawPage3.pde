// Draws page 3 of the sketch.
 
void drawPage3()
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
  
  String title = "Plot of GDP vs Pop sized by Energy Use"; 
  
  //set up scatter plot
  textSize(14);
  ch3 = new XYChart(this);
  ch3.setData(gdp_usd, pop);
  ch3.showXAxis(true);
  ch3.showYAxis(true);
  ch3.setPointSize(8);
  ch3.setPointColour(color(#342D7E,150));
  ch3.setPointSize(energy_use,25);
  ch3.setXAxisLabel("GDP-USD (Million US$ (2005))");
  ch3.setYAxisLabel("Population");
  ch3.setYFormat("###");
  ch3.setXFormat("###");
  ch3.setAxisLabelColour(0);
  ch3.setAxisValuesColour(0);
  
  ch3.draw(20,80,width*0.9,height*0.85);
  
  //title
  fill(#4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5,height*0.05);
 
}