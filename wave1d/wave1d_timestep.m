function newx=timestep(x,i,settings);
%function newx=timestep(x,i,settings)
%return (h,u) one timestep later
%take one timestep
    A=settings.A;
    B=settings.B;
    rhs=B*x; %B*x
    rhs(1)=settings.h_left(i); %left boundary, take at t+dt
    newx=A\rhs;
