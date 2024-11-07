package org.example.processoremulatorfx.Models;

import org.example.processoremulatorfx.BModels.BCpuModel;
import org.example.processoremulatorfx.BModels.BProgramModel;
import org.example.processoremulatorfx.IObserver;

import java.util.ArrayList;

public class ExecuterModel {
  CpuModel cpuModel = BCpuModel.build();
  ProgramModel programModel = BProgramModel.build();

  private int errors = 0;
  public void addErr(){ errors++; }
  public int getErrors() { return errors; }

  private boolean isEnd = false;
  private boolean isDebug = false;
  private int iter = 0;
  public int getIter() { return iter; }
  public boolean getIsEnd(){ return isEnd; }
  public boolean getIsDebug(){ return isDebug; }

  ArrayList<IObserver> allObserver = new ArrayList<>();
  void eventCall(){
    allObserver.forEach(action->action.event(this));
  }
  public void addObserver(IObserver e){
    allObserver.add(e);
    eventCall();
  }

  public void run() {
    iter = 0;
    isDebug = false;
    isEnd = false;
    for (int i = 0; iter < programModel.allCommands.size(); i++) {
      cpuModel.exec(programModel.allCommands.get(i));
      iter++;
    }
    isEnd = true;
    iter--;
    eventCall();
  }

  public void runDebug() throws Exception {
    if (!isEnd){
      if (iter < programModel.allCommands.size()){
        cpuModel.exec(programModel.allCommands.get(iter));
        isDebug = true;
        iter++;
      }
      if (iter == programModel.allCommands.size()){
        isDebug = false;
        isEnd = true;
        iter--;
      }
    }
    else
      throw new Exception("Program is end");
    eventCall();
  }

  public void reset() {
    isEnd = false;
    isDebug = false;
    iter = 0;
    cpuModel.ClearMemoryAndRegister();
    eventCall();
  }
}
