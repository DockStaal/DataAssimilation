function [times,values] = read_noos(filename);
%function [times,values] = read_noos(filename);
% format looks like
% #------------------------------------------------------
% # Timeseries retrieved from the MATROOS series database
% # Created at Mon Nov 19 12:07:20 CET 2007
% #------------------------------------------------------
% # Location    : hoekvanholland
% # Position    : (4.120118,51.979135)
% # Source      : observed
% # Unit        : waterlevel
% # Analyse time: most recent
% # Timezone    : GMT
% #------------------------------------------------------
% 200601010000   0.1200
% 200601010010   0.2000
% 200601010020   0.3100
% 200601010030   0.4300
% 200601010040   0.5700
% 200601010050   0.7300

%
% initdatstruct;
clear Series;
Series.location = '';
Series.x = 0.0;
Series.y = 0.0;
Series.source = 'matlab';
Series.unit = 'waterlevel';
Series.analysis_time = 'most recent';
Series.timezone = 'MET';

% open input file
fid = fopen(filename,'r');
if (fid<0), error(['Could not open file ',filename]);end;

%
% parse header
% 
% #------------------------------------------------------
% # Timeseries retrieved from the MATROOS series database
% # Created at Mon Nov 19 12:07:20 CET 2007
% #------------------------------------------------------
% # Location    : hoekvanholland
% # Position    : (4.120118,51.979135)
% # Source      : observed
% # Unit        : waterlevel
% # Analyse time: most recent
% # Timezone    : GMT
% #------------------------------------------------------




header = 1;
loc    = 'unknown';
pos    = [0,0];
source = 'unknown';
unit   = 'unknown';
analt  = 'unknown';
timzone= 'GMT';
while ~feof(fid) & (header==1),
    line = fgets(fid);
    index = findstr(line,'#');
    if length(index)>0,
       %disp(line);
       if length(findstr(line,'Location'))>0,
          % # Location    : hoekvanholland
          index = findstr(line,':');
          loc = line(index+1:end);
          loc = deblank(loc);
          %loc = preblank(loc);
	  Series.location = loc;
       elseif length(findstr(line,'Position'))>0,
          % # Position    : (4.120118,51.979135)
          index = findstr(line,':');
          line = line(index+1:end); %skip begin
          pos=sscanf(line,' (%f,%f)',2);
	  Series.x = pos(1);
	  Series.y = pos(2);
       elseif length(findstr(line,'Source'))>0,
          % # Source      : observed
          index = findstr(line,':');
          source = line(index+1:end);
          source=deblank(source);
          %source=preblank(source);
	  Series.source=source;
       elseif length(findstr(line,'Unit'))>0,
          % # Unit        : waterlevel
          index = findstr(line,':');
          unit = line(index+1:end);
          unit = deblank(unit);
          %unit = preblank(unit);
	  Series.unit=unit;
       elseif length(findstr(line,'Analyse'))>0,
          % # Analyse time: most recent
          index = findstr(line,':');
          analt = line(index+1:end);
          analt = deblank(analt);
          %analt = preblank(analt);
	  Series.analysis_time=analt;
       elseif length(findstr(line,'Timezone'))>0,
          % # Timezone    : GMT
          index = findstr(line,':');
          timzone = line(index+1:end);
          timzone= deblank(timzone);
          %timzone= preblank(timzone);
	  Series.timezone=timzone;
       end;
    else
       header=0; %no more header lines
    end;
end %while


%
% parse data
% 
% 200601010000   0.1200
% 200601010010   0.2000
% 200601010020   0.3100
done = 0;
nextrow = 1;
times = [];
values = [];
lineno=1;
while (done==0),
    %if (lineno<5),disp(line);end;
    lineno=lineno+1;
    dat = sscanf(line,'%f',2);
    if (length(dat>=2)),
       temp = dat(1);
       datstr = sprintf('%.0f',temp);
       year = str2num(datstr(1:4));
       month= str2num(datstr(5:6));
       day  = str2num(datstr(7:8));
       hour = str2num(datstr(9:10));
       minut= str2num(datstr(11:12));
       ctime = datenum(year,month,day,hour,minut,0);
       if (abs(dat(2))<1000), %exclude if dummy
          times(nextrow) = ctime;
          values(nextrow) = dat(2);
          nextrow=nextrow+1;
       end;
    end;
    % prepare next line
    if ~feof(fid),
       line = fgets(fid);
    else
       done=1;
    end;
end %while

%
%store data
%
Series.times = times;
Series.values = values;
fclose(fid);

