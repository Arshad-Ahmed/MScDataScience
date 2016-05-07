
void drawPage4()
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
  
  String title = "Plot of Total CO2 Emissions vs Energy Use sized by GDP"; 
  
  //set up scatter plot
  textSize(14);
  ch4 = new XYChart(this);
  ch4.setData(total_co2,energy_use);
  ch4.showXAxis(true);
  ch4.showYAxis(true);
  ch4.setPointColour(color(#FBB917,150));
  ch4.setPointSize(gdp_usd,25);
  ch4.setXAxisLabel("Total CO2 Emissions Excluding Land-Use Change and Forestry (MtCO2)");
  ch4.setYAxisLabel("Energy Use (Thous. tonnes oil eq. (ktoe))");
  ch4.setYFormat("###");
  ch4.setXFormat("###");
  ch4.setAxisLabelColour(0);
  ch4.setAxisValuesColour(0);
  
  ch4.draw(20,80,width*0.9,height*0.85);
  
  //title
  fill(#4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5,height*0.05);
}