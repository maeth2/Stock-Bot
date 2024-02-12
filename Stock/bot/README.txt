This is a Stock Bot running on CCXT

The server is Hosted on AWS.

Instructions to upload file to server:
    1. Open Filezilla.
    2. Navigate to Site Manager on the top left of the window.
    3. Under 'My Sites' click 'stock-bot' then click 'connect'.
    4. Drag and drop any files you wish to upload into the directory.

Instructions to access server:
    1. Open PuTTy.
    2. Select Stock Bot and click Load.
    3. On the left hand panel select Connection -> SSH -> Auth -> Credentials.
    4. Select "Private Key File" and open stock-bot.ppk.
    5. Click Open.

Instructions to start bot:
    1. Access server.
    2. Type:
        nohup python3 main.py &

Instructions to view bot outputs:
    1. Access server.
    2. Type 
        tail -f nohup.out

Instructions to stop bot:
    1. Access server.
    2. Type:
        ps aux | grep python3
    3. Find the port of the bot.
    4. Type "kill (port number)".
