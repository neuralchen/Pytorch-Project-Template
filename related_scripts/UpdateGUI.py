#!/usr/bin/env python3
# -*- coding:utf-8 -*-
#############################################################
# File: update_remote_project.py
# Created Date: Wednesday February 26th 2020
# Author: Chen Xuanhong
# Email: chenxuanhongzju@outlook.com
# Last Modified:  Sunday, 12th July 2020 4:33:30 pm
# Modified By: Chen Xuanhong
# Copyright (c) 2020 Shanghai Jiao Tong University
#############################################################

import os
import sys
import json
import time
import datetime
import paramiko
import threading
import tkinter as tk
from pathlib import Path



#############################################################
# Predefined functions
#############################################################

def read_config(path):
    with open(path,'r') as cf:
        nodelocaltionstr = cf.read()
        nodelocaltioninf = json.loads(nodelocaltionstr)
        if isinstance(nodelocaltioninf,str):
            nodelocaltioninf = json.loads(nodelocaltioninf)
    return nodelocaltioninf

def write_config(path, info):
    with open(path, 'w') as cf:
        configjson  = json.dumps(info, indent=4)
        cf.writelines(configjson)

class fileUploaderClass(object):
    def __init__(self,serverIp,userName,passWd,port=22):
        self.__ip__         = serverIp
        self.__userName__   = userName
        self.__passWd__     = passWd
        self.__port__       = port
        self.__ssh__        = paramiko.SSHClient()
        self.__ssh__.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    def sshScpPut(self,localFile,remoteFile):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        remoteDir  = remoteFile.split("/")
        if remoteFile[0]=='/':
            sftp.chdir('/')
            
        for item in remoteDir[0:-1]:
            if item == "":
                continue
            try:
                sftp.chdir(item)
            except:
                sftp.mkdir(item)
                sftp.chdir(item)
        sftp.put(localFile,remoteDir[-1])
        sftp.close()
        self.__ssh__.close()
        print("ssh localfile:%s remotefile:%s success"%(localFile,remoteFile))

    def sshScpGetNames(self,remoteDir):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        wocao = sftp.listdir(remoteDir)
        return wocao
    
    def sshScpGet(self, remoteFile, localFile, showProgress=False):
        self.__ssh__.connect(self.__ip__, self.__port__, self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        if showProgress:
            sftp.get(remoteFile, localFile,callback=self.__putCallBack__)
        else:
            sftp.get(remoteFile, localFile)
        sftp.close()
        self.__ssh__.close()
    
    def __putCallBack__(self,transferred,total):
        print("current transferred %.1f percent"%(transferred/total*100))
    
    def sshScpGetmd5(self, file_path):
        self.__ssh__.connect(self.__ip__, self.__port__, self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp() 
        try:
            file = sftp.open(file_path, 'rb')
            res  = (True,hashlib.new('md5', file.read()).hexdigest())
            sftp.close()
            self.__ssh__.close()
            return res
        except:
            sftp.close()
            self.__ssh__.close()
            return (False,None)
    def sshScpRename(self, oldpath, newpath):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        sftp.rename(oldpath,newpath)
        sftp.close()
        self.__ssh__.close()
        print("ssh oldpath:%s newpath:%s success"%(oldpath,newpath))

    def sshScpDelete(self,path):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        sftp.remove(path)
        sftp.close()
        self.__ssh__.close()
        print("ssh delete:%s success"%(path))
    
    def sshScpDeleteDir(self,path):
        self.__ssh__.connect(self.__ip__, self.__port__ , self.__userName__, self.__passWd__)
        sftp = paramiko.SFTPClient.from_transport(self.__ssh__.get_transport())
        sftp = self.__ssh__.open_sftp()
        self.__rm__(sftp,path)
        sftp.close()
        self.__ssh__.close()
        
    def __rm__(self,sftp,path):
        try:
            files = sftp.listdir(path=path)
            print(files)
            for f in files:
                filepath = os.path.join(path, f).replace('\\','/')
                self.__rm__(sftp,filepath)
            sftp.rmdir(path)
            print("ssh delete:%s success"%(path))
        except:
            print(path)
            sftp.remove(path)
            print("ssh delete:%s success"%(path))

#############################################################
# Main function
#############################################################

class Application(tk.Frame):
    def __init__(self, master=None):
        tk.Frame.__init__(self, master,bg='black')
        self.font_name = 'å¾®è½¯é›…é»‘'
        self.font_size = 16
        self.padx = 5
        self.pady = 5
        try:
            config = read_config("./synchronize_log.json")
        except:
            config = {
                "remote_ip": "192.168.101.57",
                "remote_user": "gdp",
                "remote_port": 22,
                "remote_passwd": "glass123456",
                "remote_path": "/home/gdp/CXH/StyleTransfer",
                "white_list": [
                    "py",
                    "yaml"
                ],
                "log_path": "./file_sync/",
                "logfilename": "filestate.json"
            }
            write_config("./synchronize_log.json",config)
            
        self.remote_ip = config["remote_ip"]
        self.remote_user = config["remote_user"]
        self.remote_port = config["remote_port"]
        self.remote_passwd = config["remote_passwd"]
        self.remote_path   = config["remote_path"]
        self.white_list   = config["white_list"]
        self.log_path = config["log_path"]
        self.logfilename= config["logfilename"]
        self.logfile_path = self.log_path + config["logfilename"]
        self.window_init()
        self.createWidgets()
    
    def window_init(self):
        self.master.title('File Synchronize')
        # self.master.bg='black'
        self.master.iconbitmap('./update.ico')
        width,height=self.master.maxsize()
        self.master.geometry("{}x{}".format(600, height//2))

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        # self.master.resizable(0,0)
    def on_closing(self):
        ssh_ip = self.ip_str.get()
        ssh_username = self.username_str.get()
        ssh_passwd = self.passwd_str.get()
        ssh_port = int(self.port_str.get())
        root_path = self.remote_path_str.get()
        white_temp = self.white_str.get()
        white_temp = white_temp.split(";")
        wocao = []
        for item in white_temp:
            if item !="":
                wocao.append(item)
        save_history = {
            "remote_ip":ssh_ip,
            "remote_user":ssh_username,
            "remote_port":ssh_port,
            "remote_passwd":ssh_passwd,
            "remote_path":root_path,
            "white_list":wocao,
            "log_path":self.log_path,
            "logfilename":self.logfilename
        }
        write_config("./synchronize_log.json",save_history)
        self.master.destroy()

    def createWidgets(self):
        font_list = (self.font_name,self.font_size)
        i = 0
        self.ip_label = tk.Label(self.master,text="Remote IP",font=font_list,fg = "black")
        self.ip_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.ip_str = tk.StringVar()
        self.ip_str.set(self.remote_ip)
        self.ip_entry = tk.Entry(self.master,font=font_list, width=30, textvariable=self.ip_str)
        self.ip_entry.grid(row=i,column=1,padx=self.padx,pady=self.pady)

        i += 1
        self.port_label = tk.Label(self.master,text="Remote Port",font=font_list,fg = "black")
        self.port_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.port_str = tk.StringVar()
        self.port_str.set(self.remote_port)
        self.port_entry = tk.Entry(self.master,font=font_list, width=30, textvariable=self.port_str)
        self.port_entry.grid(row=i,column=1,padx=self.padx,pady=self.pady)

        i += 1
        self.remote_path_label = tk.Label(self.master,text="Remote Path",font=font_list,fg = "black")
        self.remote_path_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.remote_path_str = tk.StringVar()
        self.remote_path_str.set(self.remote_path)
        self.remote_path_entry = tk.Entry(self.master,font=font_list, width=30, textvariable=self.remote_path_str)
        self.remote_path_entry.grid(row=i,column=1,padx=self.padx,pady=self.pady)

        i += 1
        self.username_label = tk.Label(self.master,text="Remote Username",font=font_list,fg = "black")
        self.username_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.username_str = tk.StringVar()
        self.username_str.set(self.remote_user)
        self.username_entry = tk.Entry(self.master,font=font_list, width=30, textvariable=self.username_str)
        self.username_entry.grid(row=i,column=1,padx=self.padx,pady=self.pady)

        i += 1
        self.passwd_label = tk.Label(self.master,text="Remote Password",font=font_list,fg = "black")
        self.passwd_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.passwd_str = tk.StringVar()
        self.passwd_str.set(self.remote_passwd)
        self.passwd_entry = tk.Entry(self.master,font=font_list, width=30, textvariable=self.passwd_str)
        self.passwd_entry.grid(row=i,column=1,padx=self.padx,pady=self.pady)

        i += 1
        self.white_label = tk.Label(self.master,text="White list",font=font_list,fg = "black")
        self.white_label.grid(row=i,padx=self.padx,pady=self.pady)
        self.white_str = tk.StringVar()
        temp_str = ";".join(self.white_list)
        self.white_str.set(temp_str)
        self.white_entry = tk.Entry(self.master,font=font_list, width=30, textvariable=self.white_str)
        self.white_entry.grid(row=i,column=1,padx=self.padx,pady=self.pady)

        i += 1
        self.run_test_button = tk.Button(self.master, width=30,text = "Synchronize files",font=font_list, command = self.Synchronize,bg='#006400', fg='#FF0000')
        self.run_test_button.grid(row=i, columnspan=2, padx=self.padx, pady=self.pady)

        i += 1
        self.text = tk.Text(self.master, wrap="word", width = 83, height=21)
        self.text.grid(row=i, columnspan=2, padx=self.padx, pady=self.pady)
        
        sys.stdout = TextRedirector(self.text, "stdout")
    
    def print(self,inputstr):
        now_str = datetime.datetime.now().strftime('%m-%d %H:%M:%S')
        print("[%s]-updated files:"%now_str,inputstr)

    def Synchronize(self):
        thread_update = threading.Thread(target=self.update)
        thread_update.start()


    def update(self):
        last_state = {}
        changed_files = []
        
        if not Path(self.log_path).exists():
            Path(self.log_path).mkdir(parents=True)
        else:
            if Path(self.logfile_path).exists():
                with open(self.logfile_path,'r') as cf:
                    nodelocaltionstr = cf.read()
                    last_state = json.loads(nodelocaltionstr)
        all_files = []
        # scan files
        white_temp = self.white_str.get()
        white_temp = white_temp.split(";")
        for item in white_temp:
            if item=="":
                self.print("something error in the white list")
                continue
            files = Path('.').glob('*.%s'%item) # ./*
            for one_file in files:
                all_files.append(one_file)
            files = Path('.').glob('*/*.%s'%item) # ./*/*
            for one_file in files:
                all_files.append(one_file)

        # check updated files
        for item in all_files:
            temp = item.stat().st_mtime
            if item._str in last_state:
                last_mtime = last_state[item._str]
                if last_mtime != temp:
                    changed_files.append(item._str)
                    last_state[item._str] = temp
            else:
                changed_files.append(item._str)
                last_state[item._str] = temp
        
        with open(self.logfile_path, 'w') as cf:
            configjson  = json.dumps(last_state, indent=4)
            cf.writelines(configjson)
        self.print(changed_files)
        ssh_ip = self.ip_str.get()
        ssh_username = self.username_str.get()
        ssh_passwd = self.passwd_str.get()
        ssh_port = int(self.port_str.get())
        root_path = self.remote_path_str.get()
        remotemachine = fileUploaderClass(ssh_ip,ssh_username,ssh_passwd,ssh_port)
        for item in changed_files:
            localfile = item
            # print("here %s"%item)
            remotefile = Path(root_path,item).as_posix()
            remotemachine.sshScpPut(localfile,remotefile)

class TextRedirector(object):
    def __init__(self, widget, tag="stdout"):
        self.widget = widget
        self.tag = tag

    def write(self, str):
        self.widget.configure(state="normal")
        self.widget.insert("end", str, (self.tag,))
        self.widget.configure(state="disabled")
        self.widget.see(tk.END)

if __name__ == "__main__":
    app = Application()
    app.mainloop()
