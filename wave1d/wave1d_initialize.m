function [x0,t0,settings_new]=wave1d_initialize(settings); 
% function [x0,t0,settings_new]=wave1d_initialize(settings); 
%return (h,u,t) at initial time 
    settings_new=settings;
    %compute initial fields and cache some things for speed
    h_0=settings.h_0;
    u_0=settings.u_0;
    n=settings.n;
    x0=zeros(2*n,1); %order h[1],u[1],...h[n],u[n]
    x0(1:2:end)=u_0(:);
    x0(2:2:end)=h_0(:);
    %time
    t=settings.t;
    reftime=settings.reftime;
    dt=settings.dt;
    times=[];
    second=1./24./60./60.; %time unit is days in matlab
    times=reftime+t*second;
    settings_new.times=times;
    %initialize coefficients
    % create matrices in form A*x_new=B*x+alpha 
    % A and B are tri-diagonal sparse matrices 
    Adata=zeros(3,2*n); %order h[1],u[1],...h[n],u[n]  
    Bdata=zeros(3,2*n);
    %left boundary
    Adata(2,1)=1.0;
    %right boundary
    Adata(2,2*n)=1.0;
    % i=2,4,6,... du/dt  + g dh/sx + f u = 0
    % m=1/2,1 1/2, ...
    %  u(n+1,m) + 0.5 g dt/dx ( h(n+1,m+1/2) - h(n+1,m-1/2)) + 0.5 dt f u(n+1,m) 
    %= u(n  ,m) - 0.5 g dt/dx ( h(n  ,m+1/2) - h(n  ,m-1/2)) - 0.5 dt f u(n  ,m)
    g=settings.g;dx=settings.dx;f=settings.f;
    temp1=0.5*g*dt/dx;
    temp2=0.5*f*dt;
    for i=2:2:(2*n-2), %in np.arange(1,2*n-1,2):
        Adata(1,i-1)= -temp1;
        Adata(2,i  )= 1.0 + temp2;
        Adata(3,i+1)= +temp1;
        Bdata(1,i-1)= +temp1;
        Bdata(2,i  )= 1.0 - temp2;
        Bdata(3,i+1)= -temp1;
    end;
    % i=3,5,7,... dh/dt + D du/dx = 0
    % m=1,2,...
    %  h(n+1,m) + 0.5 D dt/dx ( u(n+1,m+1/2) - u(n+1,m-1/2))  
    %= h(n  ,m) - 0.5 D dt/dx ( u(n  ,m+1/2) - u(n  ,m-1/2))
    D=settings.D;
    temp1=0.5*D*dt/dx;
    for i=3:2:(2*n-1), % in np.arange(2,2*n,2):
        Adata(1,i-1)= -temp1;
        Adata(2,i  )= 1.0;
        Adata(3,i+1)= +temp1;
        Bdata(1,i-1)= +temp1;
        Bdata(2,i  )= 1.0;
        Bdata(3,i+1)= -temp1;
    end;
    % build sparse matrix
    A=spdiags(Adata',[-1,0,1],2*n,2*n);
    B=spdiags(Bdata',[-1,0,1],2*n,2*n);
    settings_new.A=A; %cache for later use
    settings_new.B=B;
    t0=t(1);
