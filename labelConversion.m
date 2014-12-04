function y=labelConversion(x,flag)
% If flag=1 return label to vector
% If flag=0 returns vector to label

 if(flag==1)  
    temp=[];
    for i=1:length(x)
   
        if(x(i)==1)
         temp=[temp ;[1 0 0]];
        end   
    
        if(x(i)==2)
        temp=[temp ;[0 1 0]];
        end        
            
        if (x(i)==3)
         temp=[temp ;[0 0 1]];   
        end
        
    end            
 end
 
 if(flag==0)
      temp=[];
    for i=1:length(x)
   
        if(x(:,i)==[1 ;0; 0])
         temp=[temp 1];
           
    
        elseif(x(:,i)==[0;1;0])
        temp=[temp 2];
            
            
        elseif (x(:,i)==[0;0;1])
         temp=[temp 3];  
        
        
        else
           temp=[temp 3] ;
           %display('Ambigous')
            
         end
        
       
    end            
 end

    y=temp;

end