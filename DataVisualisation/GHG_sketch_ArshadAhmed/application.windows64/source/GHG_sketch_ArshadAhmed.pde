
//Import Libraries
import org.gicentre.utils.stat.*;    // For chart classes.
import org.gicentre.utils.colour.*;  // For colour tables.
import org.gicentre.utils.gui.*;    // For tooltips

int pageNum;    // Current page to display.
XYChart ch1,ch2,ch3, ch4,ch5,ch6,ch7;
PFont helv; //Helvetica font
PFont charter;
float[] xpos1,xpos2,xpos3,xpos4, ypo1,ypo2,ypo3,ypo4;
Tooltip tooltip1,tooltip2,tooltip3,tooltip4;
int border = 200;  // Border around each page.
color unselectedColour;

void setup()
{
  size(1000,680);
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
void draw()
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
void keyPressed()
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