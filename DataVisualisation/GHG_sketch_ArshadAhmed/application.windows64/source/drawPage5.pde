void drawPage5()
{
   fill(40,30,30);
  //fill(#34282C);
  noStroke();
  rect(20,20,width-40,height-40);
  
  textAlign(CENTER,CENTER);
  textFont(helv);
  
  fill(#F0FFFF);
  textSize(48);
  text("Instructions",width*.5,height*.15);
  
  textSize(26);
  text("This sketch uses keyboard interactions.", width*.32,height*.24);
  
  fill(#E67451);
  textSize(24);
  text("Press 1 -> CO2 Emissions vs Population sized by GDP", width*.39,height*.32);
  text("Press 2 -> C02 Emissions vs Energy Use sized by Pop", width*.39,height*.38);
  text("Press 3 -> GDP vs Pop sized by Energy Use", width*.336,height*.44);
  text("Press 4 -> Total CO2 Emissions vs Energy Use sized by GDP", width*.428,height*.5);
  text("Press 5 -> Return to Instructions Page ", width*.305,height*.56);
  text("Press 6 -> Yearly Energy Use sized by GDP ", width*.337,height*.62);
  text("Press 7 -> Yearly Total CO2 Emissions sized by GDP ", width*.388,height*.68);
  
  textSize(20);
  fill(#95B9C7);
  text("INM402 :: Data Visualisation Coursework",width*0.5+200,height-100);
  text("Arshad Ahmed",width*0.5+85,height-70);
  text("MSc Data Science 2015/16",width*0.5+140,height-40);
 
}