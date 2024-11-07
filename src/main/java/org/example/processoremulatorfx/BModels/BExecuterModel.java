package org.example.processoremulatorfx.BModels;

import org.example.processoremulatorfx.Models.ExecuterModel;

public class BExecuterModel {
  static ExecuterModel executerModel = new ExecuterModel();

  public static ExecuterModel build()  { return executerModel; }
}
