// Draws the fist page of the sketch.
Tooltip tooltip;

void drawPage1()
{
  //load data
  Table data = loadTable("CAIT_UK_SocioEconominc_GHG_CO2.csv","header,csv");
  
  //extract data columns into arrays
  float[] total_co2 = data.getFloatColumn("Total CO2 Emissions Excluding Land-Use Change and Forestry (MtCO2)");
  float[] pop = data.getFloatColumn("Population (People)"); 
  float[] gdp_usd = data.getFloatColumn("GDP-USD (Million US$ (2005))");
  float[] energy_use = data.getFloatColumn("Energy Use (Thous. tonnes oil eq. (ktoe))");
  float[] year = data.getFloatColumn("Year");
  
  String title = "Plot of Total C02 Emissions vs Population sized by GDP"; 
  
  //set up scatter plot
  textSize(14);
  ch1 = new XYChart(this);
  ch1.setData(pop, total_co2);
  ch1.showXAxis(true);
  ch1.showYAxis(true);
  ch1.setPointColour(color(#F535AA,150));
  ch1.setPointSize(gdp_usd,25);
  ch1.setXAxisLabel("Population");
  ch1.setYAxisLabel("Total CO2 Emissions Excluding Land-Use Change and Forestry (MtCO2)");
  ch1.setYFormat("###");
  ch1.setXFormat("###");
  ch1.setAxisLabelColour(0);
  ch1.setAxisValuesColour(0);
  ch1.setAxisColour(0);
  ch1.draw(20,80,width*0.9,height*0.85);
   
  //title
  fill(#4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5,height*0.05);
  rect(200,50,textWidth(title),5);
  
  
}
  
  