// Experiment after the Mondrian Project
// Algorithmic Aesthetics, Gaetan Robillard, 2022

int p = 10;

void setup(){
  size(128,128);
}

void draw(){
  background(0);
  strokeWeight(5);
  
  // vertical line
  int x = int(random(p,width-p));
  
  int cx = int(random(0,4));
  if (cx==0) {stroke(255);}
  if (cx==1) {stroke(224,184,7);}
  if (cx==2) {stroke(185,33,19);}
  if (cx==3) {stroke(54,79,179);}
  line(x,0,x,width);
  
  // horizontal line 1
  int y1 = int(random(p,height-p));
  int cy1 = int(random(0,4));
  if (cy1==0) {stroke(255);}
  if (cy1==1) {stroke(224,184,7);}
  if (cy1==2) {stroke(185,33,19);}
  if (cy1==3) {stroke(54,79,179);}
  line(0,y1,width,y1);
  
  // horizontal line 2
  float val = randomGaussian();
  int y2 = int(val*1.2 + 48);
  int cy2 = int(random(0,4));
  if (cy2==0) {stroke(255);}
  if (cy2==1) {stroke(224,184,7);}
  if (cy2==2) {stroke(185,33,19);}
  if (cy2==3) {stroke(54,79,179);}
  line(0,y2,width,y2);
  
  saveFrame("output/lines-######.jpg");
  println(frameCount);
  if (frameCount==10000){
    exit();
  }

}
