#include<stdio.h>
#include<stdlib.h>
#include<iostream>
#include <string> 
#include <time.h>
#include <unistd.h>
using namespace std;
int main () {
	
	FILE *fpt;

	fpt = fopen("MyTest.csv", "w+");
	fprintf(fpt,"src,tgt\n");
	
	srand(time(NULL));
	int R = 100;
	int L = 0;
	int a,b,c,d;
	string l,r;
	int choose= 0;
	for(int i=0;i<5000;i++)
	{
		choose= rand()%(3-0+1)+0;

		switch(choose)
		{
			case 0:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a+b-c;
				l = to_string(a);
				l +="+";
				l +=to_string(b);
				l +="-";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 1:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a+b+c;
				l = to_string(a);
				l +="+";
				l +=to_string(b);
				l +="+";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 2:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a-b-c;
				l = to_string(a);
				l +="-";
				l +=to_string(b);
				l +="-";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 3:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a-b+c;
				l = to_string(a);
				l +="-";
				l +=to_string(b);
				l +="+";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;	
		}
	}

	R = 9;
	L = 0;
	for(int i=0;i<5000;i++)
	{
		choose= rand()%(3-0+1)+0;

		switch(choose)
		{
			/*case 0:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a+b-c;
				l = to_string(a);
				l +="+";
				l +=to_string(b);
				l +="-";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 1:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a+b+c;
				l = to_string(a);
				l +="+";
				l +=to_string(b);
				l +="+";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 2:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a+b*c;
				l = to_string(a);
				l +="+";
				l +=to_string(b);
				l +="*";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 3:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-1+1)+1;
				d = a+b/c;
				l = to_string(a);
				l +="+";
				l +=to_string(b);
				l +="/";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());
				
			case 4:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a-b-c;
				l = to_string(a);
				l +="-";
				l +=to_string(b);
				l +="-";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 5:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a-b+c;
				l = to_string(a);
				l +="-";
				l +=to_string(b);
				l +="+";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 6:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a-b*c;
				l = to_string(a);
				l +="-";
				l +=to_string(b);
				l +="*";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 7:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-1+1)+1;
				d = a-b/c;
				l = to_string(a);
				l +="-";
				l +=to_string(b);
				l +="/";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());
				
			case 8:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a*b-c;
				l = to_string(a);
				l +="*";
				l +=to_string(b);
				l +="-";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 9:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a*b+c;
				l = to_string(a);
				l +="*";
				l +=to_string(b);
				l +="+";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 10:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a*b*c;
				l = to_string(a);
				l +="*";
				l +=to_string(b);
				l +="*";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 11:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-1+1)+1;
				d = a*b/c;
				l = to_string(a);
				l +="*";
				l +=to_string(b);
				l +="/";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());
				
			case 12:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-1+1)+1;
				c= rand()%(R-L+1)+L;
				d = a/b-c;
				l = to_string(a);
				l +="/";
				l +=to_string(b);
				l +="-";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 13:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-1+1)+1;
				c= rand()%(R-L+1)+L;
				d = a/b+c;
				l = to_string(a);
				l +="/";
				l +=to_string(b);
				l +="+";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 14:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-1+1)+1;
				c= rand()%(R-L+1)+L;
				d = a/b*c;
				l = to_string(a);
				l +="/";
				l +=to_string(b);
				l +="*";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 15:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-1+1)+1;
				c= rand()%(R-1+1)+1;
				d = a/b/c;
				l = to_string(a);
				l +="/";
				l +=to_string(b);
				l +="/";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());
				break;*/
			case 0:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a+b-c;
				l = to_string(a);
				l +="+";
				l +=to_string(b);
				l +="-";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 1:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a+b+c;
				l = to_string(a);
				l +="+";
				l +=to_string(b);
				l +="+";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 2:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a-b-c;
				l = to_string(a);
				l +="-";
				l +=to_string(b);
				l +="-";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;
			case 3:
				a= rand()%(R-L+1)+L;
				b= rand()%(R-L+1)+L;
				c= rand()%(R-L+1)+L;
				d = a-b+c;
				l = to_string(a);
				l +="-";
				l +=to_string(b);
				l +="+";
				l +=to_string(c);
				l +="=";
				r = to_string(d);
				fprintf(fpt,"%s,%s\n",l.c_str(),r.c_str());	
				break;		
		}
	}
	


    return 0;
}


