import processing.core.*; 
import processing.data.*; 
import processing.event.*; 
import processing.opengl.*; 

import org.gicentre.utils.stat.*; 
import org.gicentre.utils.colour.*; 
import org.gicentre.utils.gui.*; 

import java.util.HashMap; 
import java.util.ArrayList; 
import java.io.File; 
import java.io.BufferedReader; 
import java.io.PrintWriter; 
import java.io.InputStream; 
import java.io.OutputStream; 
import java.io.IOException; 

public class GHG_sketch_ArshadAhmed extends PApplet {


//Import Libraries
    // For chart classes.
  // For colour tables.
    // For tooltips

int pageNum;    // Current page to display.
XYChart ch1,ch2,ch3, ch4,ch5,ch6,ch7;
PFont helv; //Helvetica font
PFont charter;
float[] xpos1,xpos2,xpos3,xpos4, ypo1,ypo2,ypo3,ypo4;
Tooltip tooltip1,tooltip2,tooltip3,tooltip4;
int border = 200;  // Border around each page.
int unselectedColour;

public void setup()
{
  
  pageNum = 5;
  noStroke();
  unselectedColour = color(180);
  
  //Set up fonts
  helv = createFont("Helvetica",18);  
  textFont(helv);
  
  charter = createFont("Charter Bold",11);  
  textFont(charter);
   
}
 
// Selects which page to draw depending on pageNum
public void draw()
{
   background(255);
    
   if (pageNum == 1)
   {
     drawPage1();
   }
   else if (pageNum == 2)
   {
     drawPage2();
   }
   else if (pageNum == 3)
   {
     drawPage3();
   }
   else if (pageNum == 4)
   {
     drawPage4();
   }
   else if (pageNum == 5)
   {
     drawPage5();
   }
   else if (pageNum == 6)
   {
     drawPage6();
   }
    else if (pageNum == 7)
   {
     drawPage7();
   }
}
 
// Moves between pages depending in which key is pressed.
public void keyPressed()
{
 if (key=='1')
 {
   pageNum = 1;
 }
 else if (key =='2')
 {
   pageNum = 2;
 }
 else if (key =='3')
 {
   pageNum=3;
 }
 else if (key =='4')
 {
   pageNum=4;
 }
 else if (key =='5')
 {
   pageNum=5;
 }
 else if (key =='6')
 {
   pageNum=6;
 }
  else if (key =='7')
 {
   pageNum=7;
 }
}
// Draws the fist page of the sketch.
Tooltip tooltip;

public void drawPage1()
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
  ch1.setPointColour(color(0xffF535AA,150));
  ch1.setPointSize(gdp_usd,25);
  ch1.setXAxisLabel("Population");
  ch1.setYAxisLabel("Total CO2 Emissions Excluding Land-Use Change and Forestry (MtCO2)");
  ch1.setYFormat("###");
  ch1.setXFormat("###");
  ch1.setAxisLabelColour(0);
  ch1.setAxisValuesColour(0);
  ch1.setAxisColour(0);
  ch1.draw(20,80,width*0.9f,height*0.85f);
   
  //title
  fill(0xff4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5f,height*0.05f);
  
 
}
// Draws page 2 of the sketch.
 
public void drawPage2()
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
  ch2.setPointColour(color(0xff7D0552,150));
  ch2.setPointSize(pop,25);
  ch2.setXAxisLabel("Energy Use (Thous. tonnes oil eq. (ktoe))");
  ch2.setYAxisLabel("Total CO2 Emissions Excluding Land-Use Change and Forestry (MtCO2)");
  ch2.setYFormat("###");
  ch2.setXFormat("###");
  ch2.setAxisLabelColour(0);
  ch2.setAxisValuesColour(0);
  
  ch2.draw(20,80,width*0.9f,height*0.85f);
  
  //title
  fill(0xff4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5f,height*0.05f);
 
}
// Draws page 3 of the sketch.
 
public void drawPage3()
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
  ch3.setPointColour(color(0xff342D7E,150));
  ch3.setPointSize(energy_use,25);
  ch3.setXAxisLabel("GDP-USD (Million US$ (2005))");
  ch3.setYAxisLabel("Population");
  ch3.setYFormat("###");
  ch3.setXFormat("###");
  ch3.setAxisLabelColour(0);
  ch3.setAxisValuesColour(0);
  
  ch3.draw(20,80,width*0.9f,height*0.85f);
  
  //title
  fill(0xff4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5f,height*0.05f);
 
}

public void drawPage4()
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
  ch4.setPointColour(color(0xffFBB917,150));
  ch4.setPointSize(gdp_usd,25);
  ch4.setXAxisLabel("Total CO2 Emissions Excluding Land-Use Change and Forestry (MtCO2)");
  ch4.setYAxisLabel("Energy Use (Thous. tonnes oil eq. (ktoe))");
  ch4.setYFormat("###");
  ch4.setXFormat("###");
  ch4.setAxisLabelColour(0);
  ch4.setAxisValuesColour(0);
  
  ch4.draw(20,80,width*0.9f,height*0.85f);
  
  //title
  fill(0xff4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5f,height*0.05f);
}
public void drawPage5()
{
   fill(40,30,30);
  //fill(#34282C);
  noStroke();
  rect(20,20,width-40,height-40);
  
  textAlign(CENTER,CENTER);
  textFont(helv);
  
  fill(0xffF0FFFF);
  textSize(48);
  text("Instructions",width*.5f,height*.15f);
  
  textSize(26);
  text("This sketch uses keyboard interactions.", width*.32f,height*.24f);
  
  fill(0xffE67451);
  textSize(24);
  text("Press 1 -> CO2 Emissions vs Population sized by GDP", width*.39f,height*.32f);
  text("Press 2 -> C02 Emissions vs Energy Use sized by Pop", width*.39f,height*.38f);
  text("Press 3 -> GDP vs Pop sized by Energy Use", width*.336f,height*.44f);
  text("Press 4 -> Total CO2 Emissions vs Energy Use sized by GDP", width*.428f,height*.5f);
  text("Press 5 -> Return to Instructions Page ", width*.305f,height*.56f);
  text("Press 6 -> Yearly Energy Use sized by GDP ", width*.337f,height*.62f);
  text("Press 7 -> Yearly Total CO2 Emissions sized by GDP ", width*.388f,height*.68f);
  
  textSize(20);
  fill(0xff95B9C7);
  text("INM402 :: Data Visualisation Coursework",width*0.5f+200,height-100);
  text("Arshad Ahmed",width*0.5f+85,height-70);
  text("MSc Data Science 2015/16",width*0.5f+140,height-40);
 
}
public void drawPage6()
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
  ch6.setPointColour(color(0xff3B3131,150));
  ch6.setPointSize(gdp_usd,25);
  ch6.setXAxisLabel("Year");
  ch6.setYAxisLabel("Energy Use (Thous. tonnes oil eq. (ktoe))");
  ch6.setYFormat("###");
  ch6.setXFormat("###");
  ch6.setAxisLabelColour(0);
  ch6.setAxisValuesColour(0);
 
  ch6.draw(20,80,width*0.9f,height*0.85f);
  
  //title
  fill(0xff4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5f,height*0.05f);
}
public void drawPage7()
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
  ch7.setPointColour(color(0xff41A317,150));
  ch7.setPointSize(gdp_usd,25);
  ch7.setXAxisLabel("Year");
  ch7.setYAxisLabel("Total CO2 Emissions Excluding Land-Use Change and Forestry (MtCO2)");
  ch7.setYFormat("###");
  ch7.setXFormat("###");
  ch7.setAxisLabelColour(0);
  ch7.setAxisValuesColour(0);
 
  ch7.draw(20,80,width*0.9f,height*0.85f);
  
  //title
  fill(0xff4B1B4D);
  textSize(24);
  textAlign(CENTER,CENTER);
  text(title, width*0.5f,height*0.05f);
}
  public void settings() {  size(1000,680); }
  static public void main(String[] passedArgs) {
    String[] appletArgs = new String[] { "GHG_sketch_ArshadAhmed" };
    if (passedArgs != null) {
      PApplet.main(concat(appletArgs, passedArgs));
    } else {
      PApplet.main(appletArgs);
    }
  }
}
