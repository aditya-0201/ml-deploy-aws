# ml-deploy-aws
ml deploy aws






create instance of EC2
EC2

view running instance

ubuntu

click ssh,http,https

key value pair

create

add custom port
security

security group

edit

inbound rules

add rules

3000 + source(anywhere ipv4)

save rules

the connect
click connect

if it is blocked then connet through cloudshell
cloudshell

upload file

key value.pem

//permission update

chmod 400 key_value.pem

//ssh → connect to another computer

//-i key_value.pem → use your secret key

//ubuntu@ipaddress → login as "ubuntu user" to that machine

//now we are inside cloud computing

ssh -i key_value.pem ubuntu@ipaddress

sudo apt update -y

sudo apt install git, nodejs

sudo apt install npm

git clone url

go to url directory
cd mental-health

ls

npm install

node server.js

http://ip_address:3000
